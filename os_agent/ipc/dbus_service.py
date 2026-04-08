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
import threading
import time
from pathlib import Path
from threading import Thread

import dbus
import dbus.mainloop.glib
import dbus.service
from gi.repository import GLib

import yaml

from os_agent.settings import (
    SETTINGS_PATH,
    apply_to_daemon_config,
    load_settings,
)

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
        self._model_name = ""
        self._backend = "cpu"  # Updated when engine connects
        self._backend_manager = None  # Set by connect_engine()
        self._master_agent = None  # Set by connect_engine()
        self._config = {}  # Set by connect_engine()
        self._settings = {}  # User settings.yaml (set by connect_engine)
        self._config_dir = Path(
            os.environ.get("AI_DAEMON_CONFIG", "/opt/ai-daemon/config/daemon.yaml")
        ).parent
        # Lazy-load state machine
        self._load_lock = threading.Lock()
        self._loading_event = threading.Event()
        self._loading_event.set()  # "not loading" by default
        self._loading = False
        self._last_inference = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "elapsed_ms": 0,
            "cost_usd": 0.0,
            "backend": "",
        }
        log.info("D-Bus service registered at %s", DBUS_OBJECT_PATH)

    @property
    def _model_loaded(self) -> bool:
        """True iff a LocalBackend is active AND its engine is loaded."""
        if self._backend_manager is None:
            return False
        active = self._backend_manager.active
        if active is None:
            return False
        # OpenRouter "loaded" = configured; local loaded = engine.loaded
        from os_agent.inference.backend import LocalBackend
        if isinstance(active, LocalBackend):
            engine = active.engine
            return engine is not None and getattr(engine, "loaded", False)
        return True  # OpenRouter backend counts as "loaded"

    @property
    def _lazy_load_enabled(self) -> bool:
        return bool(self._settings.get("model", {}).get("lazy_load", True))

    def connect_engine(self, engine, model_name: str, backend: str,
                       config: dict | None = None,
                       settings: dict | None = None) -> None:
        """Connect the daemon to its BackendManager.

        If ``engine`` is None, the daemon starts in the not_loaded state
        (lazy-load). A MasterAgent is created only when an engine is
        actually resident — otherwise Query() / Infer() will trigger a
        load-on-demand via ``_ensure_model_loaded``.

        Args:
            engine: InferenceEngine instance, or None for not_loaded start.
            model_name: Name of the default local model (used for lazy-load).
            backend: "gpu", "cpu", or "" if not loaded yet.
            config: daemon.yaml config dict (with settings already merged).
            settings: User settings dict (from os_agent.settings.load_settings).
        """
        self._engine = engine
        self._model_name = model_name
        self._backend = backend
        self._config = config or {}
        self._settings = settings or load_settings()

        from os_agent.inference.backend import LocalBackend, BackendManager
        from os_agent.inference.model_registry import ModelRegistry
        from os_agent.agents.master import MasterAgent

        registry = ModelRegistry(self._config_dir)
        initial_backend = LocalBackend(engine) if engine is not None else None
        initial_name = model_name if engine is not None else ""
        self._backend_manager = BackendManager(
            config=self._config,
            registry=registry,
            initial_backend=initial_backend,
            initial_model_name=initial_name,
        )
        if engine is not None:
            self._master_agent = MasterAgent(engine, self._config)
            log.info("Engine connected: model=%s, backend=%s", model_name, backend)
        else:
            self._master_agent = None
            log.info("Daemon started in not_loaded state (lazy_load=%s, default=%s)",
                     self._lazy_load_enabled, model_name)

        self.StatusChanged(self._build_status())

    # ── Lazy-load helpers ───────────────────────────────────────────────────

    def _ensure_model_loaded(self) -> tuple[bool, str]:
        """Ensure a backend is loaded. Called from Query()/Infer() threads.

        Returns (ok, error). If another thread is already loading, blocks on
        the event until it completes. If we are the first caller in a
        not_loaded state, performs the load. If a backend is already active,
        this is a no-op.
        """
        if self._backend_manager is None:
            return False, "Backend manager not initialised"

        # Fast path: already loaded
        if self._backend_manager.active is not None and self._model_loaded:
            return True, ""

        # Acquire load lock — only one loader at a time
        with self._load_lock:
            # Re-check after acquiring lock — another thread may have loaded
            if self._backend_manager.active is not None and self._model_loaded:
                return True, ""

            # We are the loader
            target = self._model_name or self._settings.get(
                "model", {}).get("default_local", "")
            if not target:
                return False, "No default model configured"

            log.info("Lazy-load triggered for model: %s", target)
            self._loading = True
            self._loading_event.clear()
            self.StatusChanged(self._build_status())
            try:
                ok, err = self._backend_manager.load_local(target)
                if not ok:
                    log.error("Lazy-load failed: %s", err)
                    return False, err

                # Rebuild MasterAgent with the new engine
                from os_agent.inference.backend import LocalBackend
                from os_agent.agents.master import MasterAgent
                active = self._backend_manager.active
                if isinstance(active, LocalBackend):
                    self._engine = active.engine
                    self._master_agent = MasterAgent(self._engine, self._config)
                    self._backend = _detect_gpu_backend()
                self._model_name = target
                log.info("Lazy-load complete: %s", target)
                return True, ""
            finally:
                self._loading = False
                self._loading_event.set()
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

        if self._backend_manager is None:
            reply_cb(json.dumps({
                "error": "Engine not loaded",
                "message": "The AI daemon is starting up. Please wait a moment.",
            }))
            return

        def _run():
            try:
                # Lazy-load: if nothing is active, load the default model now.
                if self._backend_manager.active is None or not self._model_loaded:
                    ok, err = self._ensure_model_loaded()
                    if not ok:
                        reply_cb(json.dumps({
                            "error": "Load failed",
                            "message": err,
                        }))
                        return

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
                # Lazy-load: if nothing is active, load the default model now.
                if (self._backend_manager is not None
                        and (self._backend_manager.active is None
                             or not self._model_loaded)):
                    ok, err = self._ensure_model_loaded()
                    if not ok:
                        reply_cb(json.dumps({
                            "error": "Load failed",
                            "message": err,
                        }))
                        return
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
        active = self._backend_manager.active
        if active is None:
            log.error("Switch reported success but active is None")
            return False
        self._backend = active.backend_type
        from os_agent.inference.backend import LocalBackend
        if isinstance(active, LocalBackend):
            self._engine = active.engine
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
        out_signature="b",
    )
    def LoadLocalModel(self, model_name: str) -> bool:
        """Explicitly load a local GGUF into memory (Settings KCM use).

        If model_name is empty, uses the configured default. Returns True
        on success, False on failure (use GetStatus to read the error).
        """
        target = model_name or self._settings.get(
            "model", {}).get("default_local", "") or self._model_name
        if not target:
            log.error("LoadLocalModel: no model name and no default configured")
            return False
        if self._backend_manager is None:
            return False

        with self._load_lock:
            if self._loading:
                # Another load is in flight — let it finish
                pass
            self._loading = True
            self._loading_event.clear()
            self.StatusChanged(self._build_status())
            try:
                ok, err = self._backend_manager.load_local(target)
                if not ok:
                    log.error("LoadLocalModel failed: %s", err)
                    self.StatusChanged(self._build_status_with_warning(
                        f"Load failed: {err}"))
                    return False

                from os_agent.inference.backend import LocalBackend
                from os_agent.agents.master import MasterAgent
                active = self._backend_manager.active
                if isinstance(active, LocalBackend):
                    self._engine = active.engine
                    self._master_agent = MasterAgent(self._engine, self._config)
                    self._backend = _detect_gpu_backend()
                self._model_name = target
                log.info("LoadLocalModel complete: %s", target)
                return True
            finally:
                self._loading = False
                self._loading_event.set()
                self.StatusChanged(self._build_status())

    @dbus.service.method(
        DBUS_INTERFACE,
        in_signature="",
        out_signature="b",
    )
    def UnloadLocalModel(self) -> bool:
        """Unload the currently resident backend and transition to not_loaded.

        Frees VRAM/RAM. Next Query() will trigger a reload via the lazy-load
        fast-path (if lazy_load is enabled) or require an explicit
        LoadLocalModel call.
        """
        if self._backend_manager is None:
            return False
        ok, err = self._backend_manager.unload_current()
        if not ok:
            log.error("UnloadLocalModel failed: %s", err)
            return False
        self._engine = None
        self._master_agent = None
        self._backend = ""
        self.StatusChanged(self._build_status())
        return True

    @dbus.service.method(
        DBUS_INTERFACE,
        in_signature="",
        out_signature="b",
    )
    def ReloadSettings(self) -> bool:
        """Re-read ~/.config/ai-daemon/settings.yaml and apply.

        Hot-applies generation params (temperature, top_p, top_k,
        repeat_penalty, max_tokens, seed) to the active engine without a
        reload. Reload-required keys (n_ctx, n_gpu_layers) are stored in the
        merged config and will take effect on the next model load.

        Called by the Settings KCM after it writes settings.yaml.
        """
        try:
            new_settings = load_settings()
        except Exception as e:
            log.error("ReloadSettings: load failed: %s", e)
            return False

        self._settings = new_settings
        # Re-merge into the engine config so the next reload picks up
        # n_ctx / n_gpu_layers.
        base_config = _load_base_config()
        self._config = apply_to_daemon_config(base_config, new_settings)
        if self._backend_manager is not None:
            self._backend_manager.update_config(self._config)
            changed = self._backend_manager.apply_generation_params(
                new_settings.get("generation", {}))
            log.info("ReloadSettings: hot-applied %s", changed)
        self.StatusChanged(self._build_status())
        return True

    @dbus.service.method(
        DBUS_INTERFACE,
        in_signature="s",
        out_signature="b",
    )
    def SetOpenRouterKey(self, key: str) -> bool:
        """Persist the OpenRouter API key to ~/.config/ai-daemon/secrets.yaml.

        KCM calls this after storing the key in KWallet so the daemon has a
        local copy to use at runtime (openrouter.py reads secrets.yaml).
        File is written with mode 0600.

        Pass an empty string to clear the key.
        """
        from os_agent.settings import SETTINGS_DIR
        try:
            SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
            secrets_path = SETTINGS_DIR / "secrets.yaml"
            tmp = secrets_path.with_suffix(".yaml.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                yaml.safe_dump({"openrouter_api_key": key}, f)
            os.chmod(tmp, 0o600)
            os.replace(tmp, secrets_path)
            log.info("OpenRouter API key persisted (len=%d)", len(key))
            return True
        except OSError as e:
            log.error("SetOpenRouterKey failed: %s", e)
            return False

    @dbus.service.method(
        DBUS_INTERFACE,
        in_signature="ss",
        out_signature="(bs)",
        async_callbacks=("reply_cb", "error_cb"),
    )
    def TestOpenRouterConnection(self, key: str, model_id: str,
                                 reply_cb, error_cb) -> None:
        """Non-destructive OpenRouter connection test.

        Does NOT switch backends — used by the "Test Connection" button in
        the Settings KCM. Runs in a background thread (network call).
        """
        def _test():
            try:
                from os_agent.inference.openrouter import OpenRouterClient
                gen = self._settings.get("generation", {})
                client = OpenRouterClient(
                    api_key=key,
                    model_id=model_id or "deepseek/deepseek-chat",
                    temperature=gen.get("temperature", 0.5),
                    max_tokens=gen.get("max_tokens", 64),
                )
                ok, err = client.test_connection()
                client.close()
                reply_cb((dbus.Boolean(ok), dbus.String(err or "")))
            except Exception as e:
                log.error("TestOpenRouterConnection failed: %s", e)
                reply_cb((dbus.Boolean(False), dbus.String(str(e))))

        Thread(target=_test, daemon=True).start()

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

        if self._backend_manager is None:
            self.ResponseChunk(json.dumps({
                "error": "Engine not loaded",
                "message": "The AI daemon is starting up.",
            }))
            self.ResponseChunk("")  # End sentinel
            return

        def _stream():
            try:
                if self._backend_manager.active is None or not self._model_loaded:
                    ok, err = self._ensure_model_loaded()
                    if not ok:
                        self.ResponseChunk(json.dumps({
                            "error": "Load failed", "message": err}))
                        return
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

        if self._backend_manager is not None and self._backend_manager.active is not None:
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
        if self._backend == "gpu" and self._model_loaded:
            vram_used, vram_free = self._get_vram_usage()

        # Report the effective backend. If nothing is loaded we still want
        # the Plasmoid dot to reflect the *intended* backend so the user
        # knows what will happen when they click send.
        effective_backend = self._backend
        if self._backend_manager is not None and self._backend_manager.active is not None:
            effective_backend = self._backend_manager.active.backend_type
        elif not effective_backend:
            effective_backend = "cpu"

        return dbus.Dictionary({
            "model": dbus.String(self._model_name or self._settings.get(
                "model", {}).get("default_local", "")),
            "backend": dbus.String(effective_backend),
            "vram_used_mb": dbus.UInt32(vram_used),
            "vram_free_mb": dbus.UInt32(vram_free),
            "ram_used_mb": dbus.UInt32(_rss_mb()),
            "model_loaded": dbus.Boolean(self._model_loaded),
            "loading": dbus.Boolean(self._loading),
            "lazy_load": dbus.Boolean(self._lazy_load_enabled),
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


def _rss_mb() -> int:
    """Return the daemon process RSS in MB, or 0 if unavailable."""
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) // 1024
    except (OSError, ValueError):
        pass
    return 0


def _detect_gpu_backend() -> str:
    """Return 'gpu' if nvidia-smi responds, else 'cpu'."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5,
        )
        return "gpu" if result.returncode == 0 else "cpu"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "cpu"


def _load_base_config() -> dict:
    """Load the read-only daemon.yaml baked into the ISO."""
    config_path = os.environ.get(
        "AI_DAEMON_CONFIG", "/opt/ai-daemon/config/daemon.yaml"
    )
    if not Path(config_path).exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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

    # Load base config + user settings, merge them, and decide whether to
    # eagerly load the model or start in not_loaded state.
    base_config = _load_base_config()
    user_settings = load_settings()
    merged_config = apply_to_daemon_config(base_config, user_settings)

    lazy_load = bool(user_settings.get("model", {}).get("lazy_load", True))
    model_path = merged_config.get("model", {}).get("path", "")
    default_name = user_settings.get("model", {}).get(
        "default_local", Path(model_path).stem if model_path else "")

    if lazy_load:
        log.info("Lazy-load enabled — starting in not_loaded state "
                 "(default=%s, settings=%s)", default_name, SETTINGS_PATH)
        service.connect_engine(
            engine=None,
            model_name=default_name,
            backend="",
            config=merged_config,
            settings=user_settings,
        )
    elif Path(model_path).exists():
        log.info("Eager load: %s", model_path)
        try:
            from os_agent.inference.engine import InferenceEngine
            engine = InferenceEngine(merged_config)
            backend = _detect_gpu_backend()
            if backend == "cpu":
                log.info("No NVIDIA GPU detected, using CPU")
            service.connect_engine(
                engine,
                model_name=default_name or Path(model_path).stem,
                backend=backend,
                config=merged_config,
                settings=user_settings,
            )
        except Exception as e:
            log.error("Failed to load model: %s", e)
            log.info("Daemon falling back to not_loaded state")
            service.connect_engine(
                engine=None, model_name=default_name,
                backend="", config=merged_config, settings=user_settings,
            )
    else:
        log.warning("Model file not found: %s — starting in not_loaded state", model_path)
        service.connect_engine(
            engine=None, model_name=default_name,
            backend="", config=merged_config, settings=user_settings,
        )

    # Handle graceful shutdown
    loop = GLib.MainLoop()

    def shutdown(signum, frame):
        log.info("Received signal %d, shutting down...", signum)
        loop.quit()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    log.info("AI daemon ready on D-Bus: %s", DBUS_BUS_NAME)
    loop.run()
