"""D-Bus service for the AI OS daemon.

Exposes the org.aios.Daemon interface on the session bus. This is the IPC
layer between KDE Plasmoids / Settings KCM and the Python AI daemon.

Interface: org.aios.Daemon
Bus: session (user-level, not system)

Methods:
    Query(s question) → s response
        Synchronous query — sends question to the agent, returns full response.

    GetStatus() → a{sv} status
        Returns daemon status as a dictionary:
        - model (s): name of the currently loaded model
        - backend (s): "gpu", "cpu", or "openrouter"
        - vram_used_mb (u): VRAM used in MB (0 if CPU mode)
        - vram_free_mb (u): VRAM free in MB (0 if CPU mode)
        - uptime_seconds (u): daemon uptime in seconds
        - version (s): os_agent version string

    SwitchModel(s model_name) → b success
        Unloads current model, loads the named model from models.yaml.
        Returns true on success, false if model not found or load failed.

    ListModels() → aa{sv} models
        Returns array of model info dicts from models.yaml:
        - name (s): model identifier
        - path (s): filesystem path (local) or model ID (openrouter)
        - type (s): "gguf" or "openrouter"
        - description (s): human-readable description
        - active (b): whether this is the currently loaded model

Signals:
    ResponseChunk(s chunk)
        Emitted during streaming inference — one signal per text chunk.
        Plasmoid connects to this for word-by-word response display.

    StatusChanged(a{sv} status)
        Emitted when daemon status changes (model switch, mode change).

Step 6 (complete):
    - Query() routes through MasterAgent for full agent routing
    - SwitchModel() hot-swaps local GGUFs or switches to OpenRouter
    - OpenRouter: tests connection before unloading local model
    - ListModels() includes registered, discovered, and OpenRouter models
    - BackendManager handles all switching with thread safety

Remaining work for Step 7:
    - Implement QueryStreaming with actual signal emission per token
    - Connect ResponseChunk signal to QML chat UI for streaming display
    - Wire StatusChanged to update panel icon color
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from pathlib import Path
from threading import Thread

import dbus
import dbus.mainloop.glib
import dbus.service
from gi.repository import GLib

import yaml

log = logging.getLogger("ai-daemon.dbus")

# General-purpose prompt for cloud/direct backend queries (no MasterAgent routing).
# Used when OpenRouter is active — no domain classification, just general assistance.
_GENERAL_SYSTEM_PROMPT = (
    "You are an AI assistant built into a Linux-based operating system. "
    "You help users with system administration, file management, networking, "
    "package management, process management, and general Linux questions. "
    "Respond with one correct command in a bash code block followed by a one-line explanation. "
    "For conceptual questions, explain in 2-4 sentences with NO code blocks. "
    "If the request is ambiguous, ask one clarifying question. "
    "Never list alternatives. Never restate the question."
)

DBUS_INTERFACE = "org.aios.Daemon"
DBUS_BUS_NAME = "org.aios.Daemon"
DBUS_OBJECT_PATH = "/org/aios/Daemon"

# Version — matches os_agent.__main__ version
DAEMON_VERSION = "0.1.0"


class AIDaemonService(dbus.service.Object):
    """D-Bus service object exposing the org.aios.Daemon interface.

    This is a complete implementation stub. Methods that require the inference
    engine or model registry will be connected in Steps 6-7. Until then,
    they return sensible defaults or error messages so the Plasmoid can
    develop against a stable interface.
    """

    def __init__(self, bus_name: dbus.service.BusName) -> None:
        super().__init__(bus_name, DBUS_OBJECT_PATH)
        self._start_time = time.time()
        self._engine = None  # Set by connect_engine()
        self._model_name = "qwen3.5-4b-os-q4km"
        self._backend = "cpu"  # Updated when engine connects
        self._backend_manager = None  # Set by connect_engine()
        self._master_agent = None  # Set by connect_engine()
        self._config = {}  # Set by connect_engine()
        self._config_dir = Path(
            os.environ.get("AI_DAEMON_CONFIG", "/opt/ai-daemon/config/daemon.yaml")
        ).parent
        self._last_inference = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "elapsed_ms": 0,
            "cost_usd": 0.0,
            "backend": "",
        }
        log.info("D-Bus service registered at %s", DBUS_OBJECT_PATH)

    def connect_engine(self, engine, model_name: str, backend: str,
                       config: dict | None = None) -> None:
        """Connect a live InferenceEngine instance (called from daemon startup).

        Creates BackendManager (wrapping engine in LocalBackend) and
        MasterAgent for full query routing.

        Args:
            engine: InferenceEngine instance (from os_agent.inference.engine)
            model_name: Name of the loaded model
            backend: "gpu", "cpu", or "openrouter"
            config: daemon.yaml config dict
        """
        self._engine = engine
        self._model_name = model_name
        self._backend = backend
        self._config = config or {}

        from os_agent.inference.backend import LocalBackend, BackendManager
        from os_agent.inference.model_registry import ModelRegistry
        from os_agent.agents.master import MasterAgent

        local_backend = LocalBackend(engine)
        registry = ModelRegistry(self._config_dir)
        self._backend_manager = BackendManager(
            config=self._config,
            registry=registry,
            initial_backend=local_backend,
            initial_model_name=model_name,
        )
        self._master_agent = MasterAgent(engine, self._config)

        log.info("Engine connected: model=%s, backend=%s", model_name, backend)
        self.StatusChanged(self._build_status())

    # ── Methods ──────────────────────────────────────────────────────────────

    @dbus.service.method(
        DBUS_INTERFACE,
        in_signature="s",
        out_signature="s",
        async_callbacks=("reply_cb", "error_cb"),
    )
    def Query(self, question: str, reply_cb, error_cb) -> None:
        """Async query — runs inference in a background thread so the GLib
        main loop stays responsive for other D-Bus calls (GetStatus, etc.).
        """
        log.info("Query received: %s", question[:80])

        if self._master_agent is None and self._backend_manager is None:
            reply_cb(json.dumps({
                "error": "Engine not loaded",
                "message": "The AI daemon is starting up. Please wait a moment.",
            }))
            return

        def _run():
            try:
                t0 = time.time()
                if self._master_agent is not None:
                    log.info("Query routing via MasterAgent (backend=%s)", self._backend)
                    result = self._master_agent.route(question)
                    log.info("Query complete: domain=%s, %d chars",
                             result.domain, len(result.response))
                    self._update_inference_stats(int((time.time() - t0) * 1000))
                    reply_cb(result.response)
                elif self._backend_manager is not None:
                    log.info("Query routing via backend directly (backend=%s)",
                             self._backend_manager.active.backend_type)
                    response = self._backend_manager.active.infer(
                        _GENERAL_SYSTEM_PROMPT, question
                    )
                    log.info("Query complete: %d chars", len(response))
                    self._update_inference_stats(int((time.time() - t0) * 1000))
                    reply_cb(response)
            except Exception as e:
                log.error("Query failed: %s", e)
                reply_cb(json.dumps({"error": str(e)}))

        Thread(target=_run, daemon=True).start()

    @dbus.service.method(
        DBUS_INTERFACE,
        in_signature="ss",
        out_signature="s",
        async_callbacks=("reply_cb", "error_cb"),
    )
    def Infer(self, system_prompt: str, user_message: str,
              reply_cb, error_cb) -> None:
        """Raw inference — takes system prompt + user message, returns response.

        Unlike Query(), this does NOT route through MasterAgent. It sends
        the prompt directly to the active backend (local GGUF or OpenRouter).
        Used by neurosh's DaemonEngine so neurosh can do its own agent
        routing locally while using the daemon's model/backend.
        """
        log.info("Infer received: %s", user_message[:80])

        if self._backend_manager is None and self._engine is None:
            reply_cb(json.dumps({
                "error": "Engine not loaded",
                "message": "The AI daemon is starting up. Please wait a moment.",
            }))
            return

        def _run():
            try:
                t0 = time.time()
                if self._backend_manager is not None:
                    log.info("Infer via backend (type=%s)",
                             self._backend_manager.active.backend_type)
                    response = self._backend_manager.active.infer(
                        system_prompt, user_message
                    )
                elif self._engine is not None:
                    response = self._engine.infer(system_prompt, user_message)
                else:
                    response = json.dumps({"error": "No backend available"})
                elapsed_ms = int((time.time() - t0) * 1000)
                self._update_inference_stats(elapsed_ms)
                log.info("Infer complete: %d chars, %dms", len(response), elapsed_ms)
                reply_cb(response)
            except Exception as e:
                log.error("Infer failed: %s", e)
                reply_cb(json.dumps({"error": str(e)}))

        Thread(target=_run, daemon=True).start()

    @dbus.service.method(
        DBUS_INTERFACE,
        in_signature="",
        out_signature="a{sv}",
    )
    def GetLastInferenceInfo(self) -> dict:
        """Return token counts and timing from the last inference call.

        Returns dict with: prompt_tokens (u), completion_tokens (u), elapsed_ms (u).
        Used by the Plasmoid and neurosh to display tok/s and context usage.
        """
        return dbus.Dictionary({
            "prompt_tokens": dbus.UInt32(self._last_inference["prompt_tokens"]),
            "completion_tokens": dbus.UInt32(self._last_inference["completion_tokens"]),
            "elapsed_ms": dbus.UInt32(self._last_inference["elapsed_ms"]),
            "cost_usd": dbus.Double(self._last_inference.get("cost_usd", 0.0)),
            "backend": dbus.String(self._last_inference.get("backend", "")),
        }, signature="sv")

    @dbus.service.method(
        DBUS_INTERFACE,
        in_signature="",
        out_signature="a{sv}",
    )
    def GetStatus(self) -> dict:
        """Return daemon status as a D-Bus dict (a{sv})."""
        return self._build_status()

    @dbus.service.method(
        DBUS_INTERFACE,
        in_signature="s",
        out_signature="b",
    )
    def SwitchModel(self, model_name: str) -> bool:
        """Switch to a different model. Returns True on success.

        For OpenRouter: use "openrouter:<model_id>" format (e.g.
        "openrouter:deepseek/deepseek-chat"). Tests connection before
        unloading local model. On failure, keeps current model and warns.

        For local GGUF: use the model name from models.yaml.
        """
        log.info("SwitchModel requested: %s", model_name)

        if self._backend_manager is None:
            log.error("BackendManager not initialized")
            return False

        if model_name.startswith("openrouter:"):
            return self._switch_to_openrouter(model_name)

        return self._switch_to_local(model_name)

    def _switch_to_openrouter(self, model_name: str) -> bool:
        """Handle switching to OpenRouter backend."""
        model_id = model_name.split(":", 1)[1]

        from os_agent.inference.openrouter import load_api_key
        api_key = load_api_key()
        if not api_key:
            log.error("OpenRouter API key not configured")
            self.StatusChanged(self._build_status_with_warning(
                "OpenRouter API key not set. Configure in "
                "~/.config/ai-daemon/secrets.yaml or OPENROUTER_API_KEY env var."
            ))
            return False

        success, error = self._backend_manager.switch_to_openrouter(api_key, model_id)
        if not success:
            log.warning("OpenRouter switch failed: %s", error)
            self.StatusChanged(self._build_status_with_warning(
                f"OpenRouter connection failed: {error}"
            ))
            return False

        self._model_name = model_name
        self._backend = "openrouter"
        self._engine = None  # Local engine unloaded
        self._master_agent = None  # No local routing in cloud mode
        log.info("Switched to OpenRouter: %s", model_id)
        self.StatusChanged(self._build_status())
        return True

    def _switch_to_local(self, model_name: str) -> bool:
        """Handle switching to a local GGUF model."""
        success, error = self._backend_manager.switch_to_local(model_name)
        if not success:
            log.error("Local model switch failed: %s", error)
            return False

        self._model_name = model_name
        backend = self._backend_manager.active
        self._backend = backend.backend_type
        self._engine = backend.engine  # type: ignore[attr-defined]

        # Rebuild MasterAgent with the new engine
        from os_agent.agents.master import MasterAgent
        self._master_agent = MasterAgent(self._engine, self._config)

        log.info("Switched to local model: %s (backend=%s)", model_name, self._backend)
        self.StatusChanged(self._build_status())
        return True

    @dbus.service.method(
        DBUS_INTERFACE,
        in_signature="",
        out_signature="aa{sv}",
    )
    def ListModels(self) -> list:
        """List all available models with runtime state.

        Returns array of dicts with: name, path, type, description, active.
        Includes registered models, discovered GGUFs, and OpenRouter if configured.
        """
        if self._backend_manager is None:
            return dbus.Array([], signature="a{sv}")

        models = self._backend_manager.list_models()
        result = []
        for m in models:
            result.append(dbus.Dictionary({
                "name": dbus.String(m["name"]),
                "path": dbus.String(m["path"]),
                "type": dbus.String(m["type"]),
                "description": dbus.String(m["description"]),
                "active": dbus.Boolean(m["active"]),
            }, signature="sv"))

        return dbus.Array(result, signature="a{sv}")

    @dbus.service.method(
        DBUS_INTERFACE,
        in_signature="s",
        out_signature="",
    )
    def QueryStreaming(self, question: str) -> None:
        """Start streaming query — emits ResponseChunk signals.

        Routes through MasterAgent. Emits full response as single chunk
        for now. Step 7 will replace with token-by-token streaming from
        the Plasmoid.
        """
        log.info("QueryStreaming received: %s", question[:80])

        if self._master_agent is None and self._backend_manager is None:
            self.ResponseChunk(json.dumps({
                "error": "Engine not loaded",
                "message": "The AI daemon is starting up.",
            }))
            self.ResponseChunk("")  # End sentinel
            return

        def _stream():
            try:
                if self._master_agent is not None:
                    log.info("Streaming via MasterAgent (backend=%s)", self._backend)
                    result = self._master_agent.route(question)
                    self.ResponseChunk(result.response)
                elif self._backend_manager is not None:
                    log.info("Streaming via backend directly (backend=%s)",
                             self._backend_manager.active.backend_type)
                    response = self._backend_manager.active.infer(
                        _GENERAL_SYSTEM_PROMPT, question
                    )
                    self.ResponseChunk(response)
            except Exception as e:
                log.error("Streaming query failed: %s", e)
                self.ResponseChunk(json.dumps({"error": str(e)}))
            finally:
                self.ResponseChunk("")  # End sentinel

        thread = Thread(target=_stream, daemon=True)
        thread.start()

    # ── Signals ──────────────────────────────────────────────────────────────

    @dbus.service.signal(DBUS_INTERFACE, signature="s")
    def ResponseChunk(self, chunk: str) -> None:
        """Emitted per text chunk during streaming inference."""
        pass  # D-Bus framework handles emission

    @dbus.service.signal(DBUS_INTERFACE, signature="a{sv}")
    def StatusChanged(self, status: dict) -> None:
        """Emitted when daemon status changes (model switch, mode change)."""
        pass  # D-Bus framework handles emission

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _update_inference_stats(self, elapsed_ms: int) -> None:
        """Capture token counts from the active backend after inference."""
        prompt_tokens = 0
        completion_tokens = 0
        cost_usd = 0.0
        backend_type = self._backend

        if self._backend_manager is not None:
            active = self._backend_manager.active
            backend_type = active.backend_type
            if hasattr(active, 'engine') and active.engine is not None:
                # LocalBackend — llama.cpp tracks completion tokens
                completion_tokens = getattr(
                    active.engine, 'last_completion_tokens', 0)
                prompt_tokens = getattr(
                    active.engine, 'last_prompt_tokens', 0)
            elif hasattr(active, 'client'):
                # OpenRouterBackend — API returns both counts + cost
                client = active.client
                prompt_tokens = getattr(client, 'last_prompt_tokens', 0)
                completion_tokens = getattr(client, 'last_completion_tokens', 0)
                cost_usd = float(getattr(client, 'last_cost', 0.0) or 0.0)
        elif self._engine is not None:
            completion_tokens = self._engine.last_completion_tokens

        self._last_inference = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_ms": elapsed_ms,
            "cost_usd": cost_usd,
            "backend": backend_type,
        }

    def _build_status(self) -> dbus.Dictionary:
        """Build status dict for GetStatus/StatusChanged."""
        uptime = int(time.time() - self._start_time)

        vram_used = 0
        vram_free = 0
        if self._backend == "gpu":
            vram_used, vram_free = self._get_vram_usage()

        return dbus.Dictionary({
            "model": dbus.String(self._model_name),
            "backend": dbus.String(self._backend),
            "vram_used_mb": dbus.UInt32(vram_used),
            "vram_free_mb": dbus.UInt32(vram_free),
            "uptime_seconds": dbus.UInt32(uptime),
            "version": dbus.String(DAEMON_VERSION),
        }, signature="sv")

    def _build_status_with_warning(self, warning: str) -> dbus.Dictionary:
        """Build status dict with a warning message included."""
        status = self._build_status()
        status["warning"] = dbus.String(warning)
        return status

    @staticmethod
    def _get_vram_usage() -> tuple[int, int]:
        """Query nvidia-smi for VRAM usage. Returns (used_mb, free_mb)."""
        import subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 2:
                    return int(parts[0].strip()), int(parts[1].strip())
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError,
                IndexError):
            pass
        return 0, 0


def run_daemon() -> None:
    """Start the D-Bus daemon main loop.

    Called from os_agent.__main__ when --daemon flag is passed.
    """
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[ai-daemon] %(levelname)s %(message)s",
    )

    bus = dbus.SessionBus()
    bus_name = dbus.service.BusName(DBUS_BUS_NAME, bus)
    service = AIDaemonService(bus_name)

    # Load the inference engine
    config_path = os.environ.get(
        "AI_DAEMON_CONFIG", "/opt/ai-daemon/config/daemon.yaml"
    )
    if Path(config_path).exists():
        log.info("Loading config from %s", config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_path = config.get("model", {}).get("path", "")
        if Path(model_path).exists():
            log.info("Loading model: %s", model_path)
            try:
                from os_agent.inference.engine import InferenceEngine
                engine = InferenceEngine(config)  # pass dict, not path string
                n_gpu = config.get("model", {}).get("n_gpu_layers", -1)
                backend = "gpu" if n_gpu != 0 else "cpu"

                # Detect actual GPU availability
                import subprocess
                gpu_check = subprocess.run(
                    ["nvidia-smi"], capture_output=True, timeout=5,
                )
                if gpu_check.returncode != 0:
                    backend = "cpu"
                    log.info("No NVIDIA GPU detected, using CPU fallback")

                service.connect_engine(
                    engine,
                    model_name=Path(model_path).stem,
                    backend=backend,
                    config=config,
                )
            except Exception as e:
                log.error("Failed to load model: %s", e)
                log.info("Daemon running without model — queries will return errors")
        else:
            log.warning("Model file not found: %s", model_path)
            log.info("Daemon running without model")
    else:
        log.warning("Config not found: %s", config_path)

    # Handle graceful shutdown
    loop = GLib.MainLoop()

    def shutdown(signum, frame):
        log.info("Received signal %d, shutting down...", signum)
        loop.quit()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    log.info("AI daemon ready on D-Bus: %s", DBUS_BUS_NAME)
    loop.run()
