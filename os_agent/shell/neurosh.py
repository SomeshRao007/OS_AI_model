"""neurosh — Neural OS Shell.

Main REPL loop integrating the inference engine, agent framework,
memory system, and sandboxed execution into an interactive shell.
Three modes: terminal (bash), chatbot (Q&A), ai (co-pilot).
"""

from __future__ import annotations

import difflib
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml
from prompt_toolkit import PromptSession, HTML
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import DynamicCompleter
from prompt_toolkit.history import FileHistory

from prompt_toolkit.lexers import DynamicLexer

try:
    from prompt_toolkit.lexers import PygmentsLexer
    from pygments.lexers.shell import BashLexer
    _HAS_PYGMENTS = True
except ImportError:
    _HAS_PYGMENTS = False

from os_agent.agents.master import MasterAgent
from os_agent.ipc.daemon_client import DaemonEngine, daemon_is_running
from os_agent.shell.completer import create_completer, COMMON_COMMANDS
from os_agent.shell.context import EnvironmentContext
from os_agent.shell.history import ShellHistory
from os_agent.shell.modes import ModeManager, ShellMode
from os_agent.shell.renderer import Renderer, NEUROSH_STYLE
from os_agent.notify.desktop import DesktopNotifier, check_vram_and_warn
from os_agent.tools.executor import SandboxedExecutor, RiskLevel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = Path(
    os.environ.get("AI_DAEMON_CONFIG",
                   str(_PROJECT_ROOT / "os_agent" / "config" / "daemon.yaml"))
)
_HISTORY_FILE = Path.home() / ".neurosh_history"
_RC_FILE = Path.home() / ".neurosh_rc"

_log = logging.getLogger("neurosh")

# Known command names for bash detection in AI mode
_BASH_STARTERS: frozenset[str] = frozenset(
    COMMON_COMMANDS
    + [
        "git", "docker", "python", "python3", "pip", "npm", "node",
        "htop", "free", "uptime", "whoami", "id", "date", "which",
        "dpkg", "snap", "flatpak", "lsblk", "lsmod", "modprobe",
        "nslookup", "dig", "traceroute", "nc", "nft", "iptables",
        "less", "more", "vi", "vim", "nano", "clear", "history",
        "env", "export", "printenv", "mount", "umount", "lscpu",
    ]
)


# Commands that are also common English words. For these, _looks_like_bash
# requires the second token to look like a real bash argument before routing
# as a terminal command — prevents NL queries like "find files modified..." or
# "sort the output by size" from being sent to the terminal instead of the AI.
_AMBIGUOUS_COMMANDS: frozenset[str] = frozenset({
    "find", "sort", "cut", "head", "tail", "file", "host", "free",
    "diff", "stat", "watch", "kill", "cat", "mount", "rm", "top",
    "who", "id", "su", "more", "less",
})


class NeuroshShell:
    """Interactive AI-powered shell with terminal, chatbot, and AI modes."""

    def __init__(self) -> None:
        config = self._load_config()
        shell_cfg = config.get("shell", {})

        self._renderer = Renderer()

        if daemon_is_running():
            self._renderer.print_info("Connecting to AI daemon...")
            self._engine = DaemonEngine()
            self._using_daemon = True
        else:
            self._renderer.print_info("No daemon running, loading AI model locally...")
            from os_agent.inference.engine import InferenceEngine
            self._engine = InferenceEngine(config)
            self._using_daemon = False

        self._master = MasterAgent(self._engine, config)

        self._mode_mgr = ModeManager(
            default_mode=shell_cfg.get("default_mode", ShellMode.TERMINAL),
            chatbot_prefix=shell_cfg.get("chatbot_prefix", "?"),
            terminal_prefix=shell_cfg.get("terminal_prefix", "!"),
        )

        self._history = ShellHistory()

        # Initialize shell state
        self._env: dict[str, str] = os.environ.copy()
        self._aliases: dict[str, str] = {}
        self._last_exit_code = 0
        self._git_branch_cache: dict[str, tuple[float, str | None]] = {}

        # Load rc file
        self._load_rc_file()

        # Log if pygments not available
        if not _HAS_PYGMENTS:
            _log.debug("pygments not available, syntax highlighting disabled")

        # Set up prompt session with syntax highlighting and auto-suggestions
        # Note: Lexer is only applied in TERMINAL and AI modes (not CHATBOT)
        lexer = None
        if _HAS_PYGMENTS:
            _bash_lexer = PygmentsLexer(BashLexer)
            lexer = DynamicLexer(
                lambda: _bash_lexer
                if self._mode_mgr.mode in (ShellMode.TERMINAL, ShellMode.AI)
                else None
            )

        self._prompt_session = PromptSession(
            history=FileHistory(str(_HISTORY_FILE)),
            completer=DynamicCompleter(
                lambda: create_completer(self._mode_mgr.mode)
            ),
            auto_suggest=AutoSuggestFromHistory(),
            lexer=lexer,
        )

        self._env_context = EnvironmentContext()
        self._executor = SandboxedExecutor(config.get("sandbox", {}))
        self._notifier = DesktopNotifier(config.get("notify", {}))

        vram = self._engine.get_vram_usage()
        self._renderer.print_welcome(vram)

        # Show daemon status if connected via D-Bus
        if self._using_daemon and hasattr(self._engine, 'get_status'):
            status = self._engine.get_status()
            if status.get("backend") != "offline":
                self._renderer.print_info(
                    f"Connected to daemon | Model: {status.get('model', '?')} | "
                    f"Backend: {status.get('backend', '?').upper()}"
                )

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
            elif effective_mode == ShellMode.CHATBOT:
                self._handle_chatbot(cleaned)
            elif effective_mode == ShellMode.AI:
                self._handle_ai(cleaned)
            else:
                self._handle_terminal(cleaned)

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
        mode = self._mode_mgr.mode
        cwd = os.getcwd()
        home = str(Path.home())
        display_path = cwd.replace(home, "~", 1) if cwd.startswith(home) else cwd

        # Get git branch
        git_branch = self._get_git_branch(cwd)
        git_str = f" <style fg='#6272a4'>({git_branch})</style>" if git_branch else ""

        exit_code_str = ""
        if self._last_exit_code != 0:
            exit_code_str = f" <style fg='#ff5555'>✗{self._last_exit_code}</style>"

        if mode == ShellMode.CHATBOT:
            return HTML(
                f"<b>neurosh</b><style fg='#8be9fd'>[chatbot {display_path}]</style>"
                f"{git_str}{exit_code_str}&gt; "
            )
        if mode == ShellMode.AI:
            return HTML(
                f"<b>neurosh</b><style fg='#ff79c6'>[ai {display_path}]</style>"
                f"{git_str}{exit_code_str}&gt; "
            )
        return HTML(
            f"<b>neurosh</b> <style fg='#50fa7b'>[{display_path}]</style>"
            f"{git_str}{exit_code_str}&gt; "
        )

    # ── Terminal mode ────────────────────────────────────────────────────

    def _handle_terminal(self, cmd: str) -> None:
        """Execute a command via /bin/bash with inherited stdio."""
        try:
            # Check for built-ins
            if self._try_handle_builtin(cmd):
                return

            # Expand aliases
            cmd = self._expand_alias(cmd)

            result = subprocess.run(
                cmd, shell=True, executable="/bin/bash", env=self._env,
            )
            self._last_exit_code = result.returncode
            self._history.add_terminal(cmd, result.returncode)
        except Exception:
            _log.exception("Terminal command failed: %s", cmd)
            self._renderer.print_error("Command execution failed.")
            self._last_exit_code = 127

    # ── Chatbot mode ─────────────────────────────────────────────────────

    def _handle_chatbot(self, query: str) -> None:
        """Route query through the agent framework with streaming output."""
        try:
            domain = self._master.classify(query)
            agent = self._master.get_agent(domain)

            prompt = agent.augmented_prompt(query)

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
            self._update_memory(query, domain, agent, full_response)
            self._history.add_chatbot(query, domain)

        except KeyboardInterrupt:
            self._renderer.print_info("(cancelled)")
        except Exception:
            _log.exception("Chatbot query failed: %s", query)
            self._renderer.print_error("Something went wrong processing your query.")

    # ── AI co-pilot mode ─────────────────────────────────────────────────

    def _handle_ai(self, raw: str) -> None:
        """AI co-pilot: bash detection, agent routing, sandboxed execution."""
        try:
            # cd is a shell built-in — must change the parent process CWD
            if self._is_cd_command(raw):
                self._handle_cd(raw)
                return

            # Direct bash command detection (also uses built-ins path)
            if self._looks_like_bash(raw):
                self._handle_terminal(raw)
                # Record in session so subsequent AI queries know what was run
                self._master.session.add_turn(raw, "terminal", f"(ran: {raw})")
                return

            # Route through agent framework
            domain = self._master.classify(raw)
            agent = self._master.get_agent(domain)

            env_ctx = self._env_context.full_context()
            session_ctx = self._master.session.get_context_string(n=5)
            prompt = agent.augmented_prompt_with_context(raw, env_ctx, session_ctx)

            # VRAM check before inference — desktop alert if low
            check_vram_and_warn(self._engine, self._notifier)

            # Collect response silently (don't stream raw markdown to terminal)
            try:
                result = self._engine.infer_validated(prompt, raw)
            except KeyboardInterrupt:
                self._renderer.print_info("(cancelled)")
                return

            # Validator blocked the command before execution
            if result["blocked"]:
                print(f"\n[VALIDATOR] {result['error']}")
                if result["suggestion"]:
                    print(f"[SUGGESTION] {result['suggestion']}")
                return

            full_response = result["response"]
            command = result["command"]

            # Handle empty model response
            if not full_response or not full_response.strip():
                self._renderer.print_info("(no response from model)")
                return

            if not command:
                # Pure text response (conceptual answer) — print it
                print(full_response.strip())
                self._update_memory(raw, domain, agent, full_response)
                self._history.add_ai(raw, domain)
                return

            # Handle cd in extracted commands (model said "cd .." for "go back")
            if self._is_cd_command(command):
                self._handle_cd(command)
                self._update_memory(raw, domain, agent, full_response)
                self._history.add_ai(raw, domain, command)
                return

            # Risk classification
            risk = self._executor.classify_risk(command)
            in_domain = self._executor.check_domain_allowed(command, domain)
            vague = self._is_vague_query(raw)

            # Desktop notification for commands that need confirmation
            if risk != RiskLevel.SAFE or not in_domain or vague:
                self._notifier.warn_dangerous_command(command, risk, domain)

            if vague:
                # Vague query — show model's interpretation, force confirmation
                explanation = self._extract_explanation(full_response)
                if explanation:
                    print(explanation)
                self._renderer.print_risk_badge(risk, command)
                confirm = input("Proceed? [y/N] ").strip().lower()
                if confirm != "y":
                    self._renderer.print_info("Skipped.")
                    self._update_memory(raw, domain, agent, full_response)
                    self._history.add_ai(raw, domain, command)
                    return
            elif risk == RiskLevel.SAFE and in_domain:
                # Tier 1: safe + in-domain → auto-execute
                self._renderer.print_info(f"Auto-executing: {command}")
            elif not in_domain:
                # Out-of-domain → y/n with domain warning
                self._renderer.print_out_of_domain(domain, command, risk)
                confirm = input("Proceed? [y/N] ").strip().lower()
                if confirm != "y":
                    self._renderer.print_info("Skipped.")
                    self._update_memory(raw, domain, agent, full_response)
                    self._history.add_ai(raw, domain, command)
                    return
            else:
                # Tier 2/3: in-domain but moderate/dangerous → y/n
                self._renderer.print_risk_badge(risk, command)
                confirm = input("Proceed? [y/N] ").strip().lower()
                if confirm != "y":
                    self._renderer.print_info("Skipped.")
                    self._update_memory(raw, domain, agent, full_response)
                    self._history.add_ai(raw, domain, command)
                    return

            # Execute
            result = self._executor.run(command, domain=domain)
            self._renderer.print_execution_output(
                result.stdout, result.stderr, result.exit_code, result.timed_out
            )

            # Summarize long output
            if (
                result.exit_code == 0
                and result.stdout
                and len(result.stdout.splitlines()) > 50
            ):
                self._summarize_output(raw, domain, agent, result.stdout)

            self._update_memory(raw, domain, agent, full_response)
            self._history.add_ai(raw, domain, command)

        except KeyboardInterrupt:
            self._renderer.print_info("(cancelled)")
        except Exception:
            _log.exception("AI mode failed: %s", raw)
            self._renderer.print_error("Something went wrong processing your query.")

    def _looks_like_bash(self, raw: str) -> bool:
        """Heuristic: does the input start with a known command name?

        For commands that are also plain English words (find, sort, kill, etc.),
        requires the second token to look like a bash argument (a flag starting
        with -, a path starting with / ~ . , or a digit) before classifying as bash.
        This prevents natural language queries like "find files modified in the last
        24 hours" from being routed to the terminal instead of the AI.
        """
        parts = raw.split()
        if not parts:
            return False
        first = parts[0]
        basename = first.split("/")[-1]
        if basename not in _BASH_STARTERS:
            return False

        # Commands that double as English words need a bash-looking second token.
        if basename in _AMBIGUOUS_COMMANDS and len(parts) > 1:
            second = parts[1]
            if not (
                second.startswith("-")
                or second.startswith("/")
                or second.startswith("~")
                or second.startswith(".")
                or second[0].isdigit()
            ):
                return False

        return True

    @staticmethod
    def _is_cd_command(raw: str) -> bool:
        stripped = raw.strip()
        return stripped == "cd" or stripped.startswith("cd ")

    def _handle_cd(self, raw: str) -> None:
        """Handle cd as a shell built-in (changes parent process CWD)."""
        parts = raw.strip().split(maxsplit=1)
        target = parts[1] if len(parts) > 1 else "~"

        # Handle cd - (previous directory not tracked, just go home)
        if target == "-":
            prev = os.environ.get("OLDPWD", str(Path.home()))
            os.environ["OLDPWD"] = os.getcwd()
            os.chdir(prev)
            self._renderer.print_info(os.getcwd())
            return

        target = os.path.expandvars(os.path.expanduser(target))
        if not os.path.isdir(target):
            # Fuzzy match: find closest directory name in CWD
            match = self._fuzzy_find_dir(target)
            if match:
                self._renderer.print_info(f"cd: '{target}' not found, using '{match}'")
                target = match
            else:
                self._renderer.print_error(f"cd: no such directory: {target}")
                return

        os.environ["OLDPWD"] = os.getcwd()
        os.chdir(target)

    @staticmethod
    def _is_vague_query(raw: str) -> bool:
        """Detect queries too vague to auto-execute (e.g., 'show', 'check', 'info').

        A query is vague when it's very short and lacks a clear object/target.
        'show' is vague, 'show disk space' is not. 'check' is vague, 'check cpu' is not.
        """
        words = raw.strip().split()
        return len(words) <= 2 and len(raw.strip()) <= 12

    @staticmethod
    def _extract_explanation(response: str) -> str | None:
        """Extract the text explanation from a model response (strip code blocks)."""
        # Remove code blocks, keep the explanation text
        cleaned = re.sub(r"```(?:bash|sh|shell)?\n.*?```", "", response, flags=re.DOTALL)
        cleaned = cleaned.strip()
        return cleaned if cleaned else None

    @staticmethod
    def _fuzzy_find_dir(target: str) -> str | None:
        """Find the closest matching subdirectory name in CWD."""
        basename = os.path.basename(target)
        search_dir = os.path.dirname(target) or "."

        if not os.path.isdir(search_dir):
            return None

        dirs = [
            d for d in os.listdir(search_dir)
            if os.path.isdir(os.path.join(search_dir, d)) and not d.startswith(".")
        ]
        matches = difflib.get_close_matches(basename, dirs, n=1, cutoff=0.5)
        if matches:
            return os.path.join(search_dir, matches[0]) if search_dir != "." else matches[0]
        return None

    def _summarize_output(self, query: str, domain: str, agent, stdout: str) -> None:
        """For long outputs (>50 lines), ask the agent to summarize."""
        # Truncate to fit context window (~300 tokens = ~1200 chars)
        truncated = stdout[:1200]
        summary_query = f"Summarize this output for '{query}':\n{truncated}"
        env_ctx = self._env_context.cwd_context()
        prompt = agent.augmented_prompt_with_context(summary_query, env_ctx)

        self._renderer.print_info("[AI Summary]")
        try:
            for token in self._engine.infer_streaming(prompt, summary_query):
                sys.stdout.write(token)
                sys.stdout.flush()
        except KeyboardInterrupt:
            pass
        finally:
            sys.stdout.write("\n")
            sys.stdout.flush()

    # ── Built-ins, aliases, and rc file ─────────────────────────────────

    def _load_rc_file(self) -> None:
        """Load rc file at startup and apply aliases/exports."""
        if not _RC_FILE.exists():
            return

        try:
            with open(_RC_FILE) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    if line.startswith("alias "):
                        self._parse_alias_line(line[6:])
                    elif line.startswith("export "):
                        self._parse_export_line(line[7:])
                    elif line.startswith("unalias "):
                        parts = line[8:].split()
                        if parts:
                            self._aliases.pop(parts[0], None)
                    elif line.startswith("unset "):
                        parts = line[6:].split()
                        if parts:
                            self._env.pop(parts[0], None)
        except Exception as e:
            _log.debug("rc file load failed: %s", e)

    def _try_handle_builtin(self, cmd: str) -> bool:
        """Try to handle built-in commands. Returns True if handled."""
        parts = cmd.split(maxsplit=1)
        if not parts:
            return False

        builtin = parts[0]

        if builtin == "export":
            if len(parts) > 1:
                self._parse_export_line(parts[1])
            else:
                # Bare export: list all
                for k, v in self._env.items():
                    if k not in os.environ or os.environ.get(k) != v:
                        print(f"export {k}='{v}'")
            return True

        if builtin == "unset":
            if len(parts) > 1:
                for var in parts[1].split():
                    self._env.pop(var, None)
            return True

        if builtin == "alias":
            if len(parts) > 1:
                self._parse_alias_line(parts[1])
            else:
                # Bare alias: list all
                for name, cmd_str in self._aliases.items():
                    print(f"alias {name}='{cmd_str}'")
            return True

        if builtin == "unalias":
            if len(parts) > 1:
                for name in parts[1].split():
                    self._aliases.pop(name, None)
            return True

        return False

    def _parse_export_line(self, line: str) -> None:
        """Parse 'KEY=VALUE KEY2=VALUE2' or 'KEY' and update env."""
        # Use shlex for proper quoting
        try:
            tokens = shlex.split(line)
        except ValueError:
            tokens = line.split()

        for token in tokens:
            if "=" in token:
                key, val = token.split("=", 1)
                self._env[key] = val
            else:
                # export KEY — read from current env
                if token in os.environ:
                    self._env[token] = os.environ[token]

    def _parse_alias_line(self, line: str) -> None:
        """Parse 'NAME=cmd' or 'NAME=\"cmd\"' and update aliases."""
        if "=" not in line:
            return

        name, cmd_str = line.split("=", 1)
        name = name.strip()

        # Remove quotes
        if cmd_str and cmd_str[0] in ("'", '"') and cmd_str[-1] == cmd_str[0]:
            cmd_str = cmd_str[1:-1]

        self._aliases[name] = cmd_str

    def _expand_alias(self, cmd: str) -> str:
        """Expand the first token if it's an alias."""
        parts = cmd.split(maxsplit=1)
        if not parts:
            return cmd

        first = parts[0]
        if first in self._aliases:
            rest = parts[1] if len(parts) > 1 else ""
            return self._aliases[first] + (" " + rest if rest else "")

        return cmd

    def _get_git_branch(self, cwd: str) -> str | None:
        """Get git branch by reading .git/HEAD directly (with 2s cache)."""
        now = time.time()
        cached = self._git_branch_cache.get(cwd)
        if cached and now - cached[0] < 2:
            return cached[1]

        # Walk up to find .git
        current = Path(cwd)
        while current != current.parent:
            git_head = current / ".git" / "HEAD"
            if git_head.exists():
                try:
                    content = git_head.read_text().strip()
                    if content.startswith("ref: refs/heads/"):
                        branch = content[16:]
                        self._git_branch_cache[cwd] = (now, branch)
                        return branch
                    else:
                        # Detached HEAD
                        self._git_branch_cache[cwd] = (now, None)
                        return None
                except Exception:
                    pass
            current = current.parent

        self._git_branch_cache[cwd] = (now, None)
        return None

    # ── Shared helpers ───────────────────────────────────────────────────

    def _update_memory(self, query: str, domain: str, agent, response: str) -> None:
        """Update all memory tiers after an agent response."""
        if agent._memory and response:
            agent._memory.store(query, response)
        self._master.session.add_turn(query, domain, response, [])
        self._master.shared_state.log_action(domain, query, response[:100])

    # ── Meta commands ────────────────────────────────────────────────────

    def _handle_meta(self, cmd: str) -> bool:
        """Handle / commands. Returns True if shell should exit."""
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0].lower()

        handlers = {
            "/chatbot": self._cmd_chatbot,
            "/terminal": self._cmd_terminal,
            "/ai": self._cmd_ai,
            "/status": self._cmd_status,
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

    def _cmd_chatbot(self) -> bool:
        self._mode_mgr.switch_to_chatbot()
        self._renderer.print_info("Switched to chatbot mode.")
        return False

    def _cmd_terminal(self) -> bool:
        self._mode_mgr.switch_to_terminal()
        self._renderer.print_info("Switched to terminal mode.")
        return False

    def _cmd_ai(self) -> bool:
        self._mode_mgr.switch_to_ai()
        self._renderer.print_info("Switched to AI co-pilot mode.")
        return False

    def _cmd_status(self) -> bool:
        """Show daemon status, model info, and last inference stats."""
        if not self._using_daemon or not hasattr(self._engine, 'get_status'):
            self._renderer.print_info("Running locally (no daemon connection)")
            vram = self._engine.get_vram_usage()
            if vram.get("total", 0) > 0:
                self._renderer.print_info(
                    f"  VRAM: {vram['used']}/{vram['total']} MB"
                )
            return False

        status = self._engine.get_status()
        lines = []
        lines.append(f"  Model:   {status.get('model', 'unknown')}")
        lines.append(f"  Backend: {status.get('backend', 'offline').upper()}")

        vram_used = int(status.get("vram_used_mb", 0))
        vram_free = int(status.get("vram_free_mb", 0))
        if vram_used > 0:
            lines.append(f"  VRAM:    {vram_used}/{vram_used + vram_free} MB")

        uptime = int(status.get("uptime_seconds", 0))
        if uptime > 0:
            h, m = divmod(uptime // 60, 60)
            lines.append(f"  Uptime:  {h}h {m}m" if h else f"  Uptime:  {m}m")

        info = self._engine.get_last_inference_info()
        pt = info.get("prompt_tokens", 0)
        ct = info.get("completion_tokens", 0)
        ms = info.get("elapsed_ms", 0)
        if ct > 0:
            tok_s = ct / (ms / 1000) if ms > 0 else 0
            lines.append(f"  Last:    prompt={pt} completion={ct} "
                         f"({tok_s:.1f} tok/s)")

        self._renderer.print_meta_response("Daemon Status", "\n".join(lines))
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
