"""D-Bus client for neurosh — routes inference through the running daemon.

Instead of loading a local model (3-4 GB RAM), neurosh connects to the
ai-daemon over D-Bus and uses whatever backend is active (local GGUF,
CPU fallback, or OpenRouter). When the daemon switches models, neurosh
automatically uses the new one.

Provides the same infer()/infer_streaming()/infer_validated()/infer_with_rag()
interface as InferenceEngine so MasterAgent and specialist agents work
transparently.
"""

from __future__ import annotations

import json
import logging
from typing import Generator

import dbus

log = logging.getLogger("neurosh.daemon-client")

DBUS_BUS_NAME = "org.aios.Daemon"
DBUS_OBJECT_PATH = "/org/aios/Daemon"
DBUS_INTERFACE = "org.aios.Daemon"


def daemon_is_running() -> bool:
    """Check if the ai-daemon is registered on the session bus."""
    try:
        bus = dbus.SessionBus()
        proxy = bus.get_object("org.freedesktop.DBus", "/org/freedesktop/DBus")
        names = proxy.ListNames(dbus_interface="org.freedesktop.DBus")
        return DBUS_BUS_NAME in names
    except dbus.exceptions.DBusException:
        return False


class DaemonEngine:
    """D-Bus proxy that looks like InferenceEngine to the agent framework.

    Methods mirror InferenceEngine so MasterAgent, BaseAgent.handle(), and
    neurosh can use this as a drop-in replacement. Inference goes to the
    daemon; RAG and validation run locally in neurosh's process.
    """

    def __init__(self) -> None:
        bus = dbus.SessionBus()
        self._proxy = bus.get_object(DBUS_BUS_NAME, DBUS_OBJECT_PATH)
        self._iface = dbus.Interface(self._proxy, DBUS_INTERFACE)
        self._last_completion_tokens = 0
        log.info("Connected to ai-daemon via D-Bus")

    @property
    def loaded(self) -> bool:
        """Always True — the daemon manages model lifecycle."""
        return True

    @property
    def last_completion_tokens(self) -> int:
        return self._last_completion_tokens

    def infer(self, system_prompt: str, user_message: str,
              max_tokens: int | None = None) -> str:
        """Synchronous inference via D-Bus daemon.

        Calls the daemon's Infer(system_prompt, user_message) method which
        sends the prompt directly to the active backend (local GGUF or
        OpenRouter) without re-routing through MasterAgent.
        """
        try:
            # CPU inference can take 30-60s; default D-Bus timeout is 25s.
            # Set 120s timeout to avoid NoReply errors.
            response = self._iface.Infer(
                system_prompt, user_message,
                timeout=120,
            )
        except dbus.exceptions.DBusException as e:
            log.error("D-Bus Infer failed: %s", e)
            return f"Error: AI daemon unavailable ({e.get_dbus_name()})"

        # Check for JSON error responses from the daemon
        if response.startswith("{"):
            try:
                data = json.loads(response)
                if "error" in data:
                    return f"Error: {data.get('message', data['error'])}"
            except (json.JSONDecodeError, KeyError):
                pass

        return response

    def infer_streaming(self, system_prompt: str, user_message: str,
                        max_tokens: int | None = None) -> Generator[str, None, None]:
        """Streaming inference — falls back to synchronous for now.

        True token-by-token streaming via D-Bus signals is Step 7 scope.
        For now, get the full response and yield it as one chunk.
        """
        response = self.infer(system_prompt, user_message, max_tokens)
        yield response

    def infer_with_rag(self, system_prompt: str, query: str,
                       max_tokens: int | None = None) -> str:
        """RAG-augmented inference — RAG runs locally, inference via daemon."""
        from os_agent.inference.rag import build_rag_context

        rag_ctx = build_rag_context(query)
        if rag_ctx:
            system_prompt = system_prompt + f"\n\nCOMMAND REFERENCE: {rag_ctx}"
        return self.infer(system_prompt, query, max_tokens)

    def infer_validated(self, system_prompt: str, query: str,
                        max_tokens: int | None = None) -> dict:
        """RAG + validate — same logic as InferenceEngine, inference via daemon."""
        from os_agent.tools.parser import extract_command
        from os_agent.inference.validator import validate

        response = self.infer_with_rag(system_prompt, query, max_tokens)
        command = extract_command(response)

        if command:
            result = validate(command)
            if not result["ok"]:
                return {
                    "response": response,
                    "command": command,
                    "blocked": True,
                    "error": result["error"],
                    "suggestion": result.get("suggestion", ""),
                }

        return {"response": response, "command": command, "blocked": False}

    def get_vram_usage(self) -> dict[str, int]:
        """Query VRAM via D-Bus GetStatus instead of nvidia-smi directly."""
        try:
            status = self._iface.GetStatus()
            return {
                "used": int(status.get("vram_used_mb", 0)),
                "total": int(status.get("vram_used_mb", 0) + status.get("vram_free_mb", 0)),
                "free": int(status.get("vram_free_mb", 0)),
            }
        except dbus.exceptions.DBusException:
            return {"used": 0, "total": 0, "free": 0}

    def get_status(self) -> dict:
        """Get daemon status (model, backend, VRAM, uptime)."""
        try:
            status = self._iface.GetStatus()
            return {k: str(v) for k, v in status.items()}
        except dbus.exceptions.DBusException:
            return {"backend": "offline", "model": "none"}

    def get_last_inference_info(self) -> dict:
        """Get token counts and timing from the last inference."""
        try:
            info = self._iface.GetLastInferenceInfo()
            return {
                "prompt_tokens": int(info.get("prompt_tokens", 0)),
                "completion_tokens": int(info.get("completion_tokens", 0)),
                "elapsed_ms": int(info.get("elapsed_ms", 0)),
            }
        except dbus.exceptions.DBusException:
            return {"prompt_tokens": 0, "completion_tokens": 0, "elapsed_ms": 0}

    def unload(self) -> None:
        """No-op — daemon manages model lifecycle, not neurosh."""
        pass
