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

Remaining work for Steps 6-7:
    - Step 6 (Model Registry): Implement SwitchModel with actual model
      hot-swap logic via InferenceEngine. Implement ListModels reading
      from models.yaml. Connect BackendManager for GPU→CPU→OpenRouter
      fallback chain.
    - Step 7 (Plasmoid): Implement QueryStreaming with actual signal
      emission per token. Connect ResponseChunk signal to QML chat UI
      for streaming display. Wire StatusChanged to update panel icon color.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from threading import Thread

import dbus
import dbus.mainloop.glib
import dbus.service
from gi.repository import GLib

import yaml

log = logging.getLogger("ai-daemon.dbus")

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
        self._config_dir = Path(
            os.environ.get("AI_DAEMON_CONFIG", "/opt/ai-daemon/config/daemon.yaml")
        ).parent
        log.info("D-Bus service registered at %s", DBUS_OBJECT_PATH)

    def connect_engine(self, engine, model_name: str, backend: str) -> None:
        """Connect a live InferenceEngine instance (called from daemon startup).

        Args:
            engine: InferenceEngine instance (from os_agent.inference.engine)
            model_name: Name of the loaded model
            backend: "gpu", "cpu", or "openrouter"
        """
        self._engine = engine
        self._model_name = model_name
        self._backend = backend
        log.info("Engine connected: model=%s, backend=%s", model_name, backend)
        self.StatusChanged(self._build_status())

    # ── Methods ──────────────────────────────────────────────────────────────

    @dbus.service.method(
        DBUS_INTERFACE,
        in_signature="s",
        out_signature="s",
    )
    def Query(self, question: str) -> str:
        """Synchronous query — full response returned as string."""
        log.info("Query received: %s", question[:80])

        if self._engine is None:
            return json.dumps({
                "error": "Engine not loaded",
                "message": "The AI daemon is starting up. Please wait a moment.",
            })

        # Delegate to the agent framework
        # Step 6 will wire this through MasterAgent for full routing
        try:
            response = self._engine.generate(question)
            return response
        except Exception as e:
            log.error("Query failed: %s", e)
            return json.dumps({"error": str(e)})

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

        Step 6 implementation will:
        1. Look up model_name in models.yaml
        2. Unload current model from VRAM
        3. Load new model
        4. Update self._model_name and self._backend
        5. Emit StatusChanged signal
        """
        log.info("SwitchModel requested: %s", model_name)

        # Load models.yaml to validate the requested model exists
        models_path = self._config_dir / "models.yaml"
        if not models_path.exists():
            log.error("models.yaml not found at %s", models_path)
            return False

        with open(models_path) as f:
            registry = yaml.safe_load(f)

        local_models = registry.get("models", {}).get("local", [])
        match = None
        for m in local_models:
            if m.get("name") == model_name:
                match = m
                break

        if match is None:
            log.error("Model '%s' not found in registry", model_name)
            return False

        model_path = match.get("path", "")
        if not Path(model_path).exists():
            log.error("Model file not found: %s", model_path)
            return False

        # Step 6: actual hot-swap logic goes here
        # self._engine.unload()
        # self._engine.load(model_path)
        # self._model_name = model_name
        # self.StatusChanged(self._build_status())
        log.info("SwitchModel: model '%s' validated, hot-swap deferred to Step 6", model_name)
        return False  # Return False until Step 6 implements actual swap

    @dbus.service.method(
        DBUS_INTERFACE,
        in_signature="",
        out_signature="aa{sv}",
    )
    def ListModels(self) -> list:
        """List all available models from the registry.

        Returns array of dicts with: name, path, type, description, active.
        """
        models_path = self._config_dir / "models.yaml"
        if not models_path.exists():
            return []

        with open(models_path) as f:
            registry = yaml.safe_load(f)

        result = []
        for m in registry.get("models", {}).get("local", []):
            result.append(dbus.Dictionary({
                "name": dbus.String(m.get("name", "")),
                "path": dbus.String(m.get("path", "")),
                "type": dbus.String(m.get("type", "gguf")),
                "description": dbus.String(m.get("description", "")),
                "active": dbus.Boolean(m.get("name") == self._model_name),
            }, signature="sv"))

        # Include OpenRouter if configured
        openrouter = registry.get("models", {}).get("openrouter", {})
        if openrouter.get("enabled"):
            result.append(dbus.Dictionary({
                "name": dbus.String(openrouter.get("default_model", "openrouter")),
                "path": dbus.String("https://openrouter.ai/api/v1"),
                "type": dbus.String("openrouter"),
                "description": dbus.String("Cloud inference via OpenRouter API"),
                "active": dbus.Boolean(self._backend == "openrouter"),
            }, signature="sv"))

        return dbus.Array(result, signature="a{sv}")

    @dbus.service.method(
        DBUS_INTERFACE,
        in_signature="s",
        out_signature="",
    )
    def QueryStreaming(self, question: str) -> None:
        """Start streaming query — emits ResponseChunk signals.

        Step 7 implementation will:
        1. Run inference in a background thread
        2. Emit ResponseChunk(chunk) for each generated token
        3. Emit ResponseChunk("") as end-of-stream sentinel
        """
        log.info("QueryStreaming received: %s", question[:80])

        if self._engine is None:
            self.ResponseChunk(json.dumps({
                "error": "Engine not loaded",
                "message": "The AI daemon is starting up.",
            }))
            self.ResponseChunk("")  # End sentinel
            return

        def _stream():
            try:
                # Step 7: replace with actual token-by-token streaming
                # For now, generate full response and emit as single chunk
                response = self._engine.generate(question)
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
                return int(parts[0].strip()), int(parts[1].strip())
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
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
