"""neurosh — Neural OS Shell.

Main REPL loop integrating the inference engine, agent framework,
and memory system into an interactive shell experience.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import yaml
from prompt_toolkit import PromptSession, HTML
from prompt_toolkit.completion import DynamicCompleter
from prompt_toolkit.history import FileHistory

from os_agent.inference.engine import InferenceEngine
from os_agent.agents.master import MasterAgent
from os_agent.shell.completer import create_completer
from os_agent.shell.history import ShellHistory
from os_agent.shell.modes import ModeManager, ShellMode
from os_agent.shell.renderer import Renderer, NEUROSH_STYLE

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "os_agent" / "config" / "daemon.yaml"
_HISTORY_FILE = Path.home() / ".neurosh_history"

_log = logging.getLogger("neurosh")


class NeuroshShell:
    """Interactive AI-powered shell with direct and AI modes."""

    def __init__(self) -> None:
        config = self._load_config()
        shell_cfg = config.get("shell", {})

        self._renderer = Renderer()
        self._renderer.print_info("Loading AI model...")

        self._engine = InferenceEngine(config)
        self._master = MasterAgent(self._engine, config)

        self._mode_mgr = ModeManager(
            default_mode=shell_cfg.get("default_mode", ShellMode.DIRECT),
            ai_prefix=shell_cfg.get("ai_prefix", "?"),
            direct_prefix=shell_cfg.get("direct_prefix", "!"),
        )

        self._history = ShellHistory()
        self._prompt_session = PromptSession(
            history=FileHistory(str(_HISTORY_FILE)),
            completer=DynamicCompleter(
                lambda: create_completer(self._mode_mgr.mode)
            ),
        )

        vram = self._engine.get_vram_usage()
        self._renderer.print_welcome(vram)

    def run(self) -> None:
        """Main REPL loop."""
        while True:
            prompt_html = self._build_prompt()
            raw = self._read_input(prompt_html)
            if raw is None:
                break

            raw = raw.strip()
            if not raw:
                continue

            effective_mode, cleaned = self._mode_mgr.classify_input(raw)

            if effective_mode == "meta":
                should_exit = self._handle_meta(cleaned)
                if should_exit:
                    break
            elif effective_mode == ShellMode.AI:
                self._handle_ai(cleaned)
            else:
                self._handle_direct(cleaned)

        self._renderer.print_info("Goodbye.")

    # ── Input ────────────────────────────────────────────────────────────

    def _read_input(self, prompt_html: HTML) -> str | None:
        """Read one line from the user. Returns None on EOF (Ctrl+D)."""
        while True:
            try:
                return self._prompt_session.prompt(prompt_html, style=NEUROSH_STYLE)
            except KeyboardInterrupt:
                continue
            except EOFError:
                return None

    def _build_prompt(self) -> HTML:
        if self._mode_mgr.mode == ShellMode.AI:
            return HTML("<b>neurosh</b><style fg='#61afef'>[ai]</style>&gt; ")
        return HTML("<b>neurosh</b>&gt; ")

    # ── Direct mode ──────────────────────────────────────────────────────

    def _handle_direct(self, cmd: str) -> None:
        """Execute a command via /bin/bash with inherited stdio."""
        try:
            result = subprocess.run(
                cmd, shell=True, executable="/bin/bash",
            )
            self._history.add_direct(cmd, result.returncode)
        except Exception:
            _log.exception("Direct command failed: %s", cmd)
            self._renderer.print_error("Command execution failed.")

    # ── AI mode ──────────────────────────────────────────────────────────

    def _handle_ai(self, query: str) -> None:
        """Route query through the agent framework with streaming output."""
        try:
            domain = self._master.classify(query)
            agent = self._master.get_agent(domain)
            self._renderer.print_domain_badge(domain)

            # Search FAISS memory for similar past solutions
            hits = []
            if agent._memory:
                hits = agent._memory.search(query, top_k=3)

            prompt = agent.augmented_prompt(query)

            # Stream tokens to terminal
            tokens: list[str] = []
            try:
                for token in self._engine.infer_streaming(prompt, query):
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    tokens.append(token)
            except KeyboardInterrupt:
                self._renderer.print_info("\n(cancelled)")
                return
            finally:
                sys.stdout.write("\n")
                sys.stdout.flush()

            full_response = "".join(tokens)

            # Update memory tiers (mirrors MasterAgent.route() logic)
            if agent._memory and full_response:
                agent._memory.store(query, full_response)
            hit_texts = [h.response for h in hits]
            self._master.session.add_turn(query, domain, full_response, hit_texts)
            self._master.shared_state.log_action(domain, query, full_response[:100])

            self._history.add_ai(query, domain)

        except KeyboardInterrupt:
            self._renderer.print_info("(cancelled)")
        except Exception:
            _log.exception("AI query failed: %s", query)
            self._renderer.print_error("Something went wrong processing your query.")

    # ── Meta commands ────────────────────────────────────────────────────

    def _handle_meta(self, cmd: str) -> bool:
        """Handle / commands. Returns True if shell should exit."""
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0].lower()

        handlers = {
            "/ai": self._cmd_ai,
            "/direct": self._cmd_direct,
            "/history": self._cmd_history,
            "/memory": self._cmd_memory,
            "/agents": self._cmd_agents,
            "/clear": self._cmd_clear,
            "/help": self._cmd_help,
            "/exit": self._cmd_exit,
            "/quit": self._cmd_exit,
        }

        handler = handlers.get(command)
        if handler is None:
            self._renderer.print_error(f"Unknown command: {command}")
            return False

        return handler()

    def _cmd_ai(self) -> bool:
        self._mode_mgr.switch_to_ai()
        self._renderer.print_info("Switched to AI mode.")
        return False

    def _cmd_direct(self) -> bool:
        self._mode_mgr.switch_to_direct()
        self._renderer.print_info("Switched to direct mode.")
        return False

    def _cmd_history(self) -> bool:
        self._renderer.print_meta_response(
            "Command History", self._history.format_display(20)
        )
        return False

    def _cmd_memory(self) -> bool:
        lines = []
        for domain in ("files", "network", "process", "packages", "kernel"):
            agent = self._master.get_agent(domain)
            if agent._memory and agent._memory._index is not None:
                count = agent._memory._index.ntotal
                lines.append(f"  {domain:12s}  {count} stored solutions")
            else:
                lines.append(f"  {domain:12s}  (no memory loaded)")
        self._renderer.print_meta_response("Agent Memory", "\n".join(lines))
        return False

    def _cmd_agents(self) -> bool:
        turn_count = self._master.session.turn_count
        lines = [f"  Session turns: {turn_count}", ""]
        for domain in ("files", "network", "process", "packages", "kernel"):
            lines.append(f"  {domain:12s}  active")
        self._renderer.print_meta_response("Active Agents", "\n".join(lines))
        return False

    def _cmd_clear(self) -> bool:
        self._master.session.clear()
        self._renderer.print_info("Session context cleared.")
        return False

    def _cmd_help(self) -> bool:
        self._renderer.print_help()
        return False

    def _cmd_exit(self) -> bool:
        return True

    # ── Config ───────────────────────────────────────────────────────────

    @staticmethod
    def _load_config() -> dict:
        if _CONFIG_PATH.exists():
            return yaml.safe_load(_CONFIG_PATH.read_text()) or {}
        return {}
