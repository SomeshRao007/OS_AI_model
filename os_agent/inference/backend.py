"""Unified backend interface — abstracts local GGUF and OpenRouter inference.

Agents call backend.infer() / backend.infer_streaming() without caring
whether inference happens locally or in the cloud. BackendManager handles
switching between backends and model hot-swap.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator

log = logging.getLogger("ai-daemon.backend")


class InferenceBackend(ABC):
    """Abstract base for all inference backends."""

    @abstractmethod
    def infer(self, system_prompt: str, user_message: str,
              max_tokens: int | None = None) -> str:
        """Synchronous inference. Returns cleaned response text."""

    @abstractmethod
    def infer_streaming(self, system_prompt: str, user_message: str,
                        max_tokens: int | None = None) -> Generator[str, None, None]:
        """Streaming inference. Yields text chunks."""

    @abstractmethod
    def unload(self) -> None:
        """Free resources (VRAM/RAM for local, no-op for cloud)."""

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Return 'gpu', 'cpu', or 'openrouter'."""


class LocalBackend(InferenceBackend):
    """Wraps InferenceEngine for local GGUF model inference."""

    def __init__(self, engine) -> None:
        """Wrap an existing InferenceEngine instance.

        Args:
            engine: os_agent.inference.engine.InferenceEngine instance.
        """
        self._engine = engine

    @property
    def engine(self):
        """Direct access to the underlying InferenceEngine.

        Used by MasterAgent and specialist agents that call engine.infer()
        or engine.infer_with_rag() directly.
        """
        return self._engine

    def infer(self, system_prompt: str, user_message: str,
              max_tokens: int | None = None) -> str:
        return self._engine.infer(system_prompt, user_message, max_tokens)

    def infer_streaming(self, system_prompt: str, user_message: str,
                        max_tokens: int | None = None) -> Generator[str, None, None]:
        yield from self._engine.infer_streaming(system_prompt, user_message, max_tokens)

    def infer_with_rag(self, system_prompt: str, query: str,
                       max_tokens: int | None = None) -> str:
        """Delegate RAG-augmented inference to the engine."""
        return self._engine.infer_with_rag(system_prompt, query, max_tokens)

    def infer_validated(self, system_prompt: str, query: str,
                        max_tokens: int | None = None) -> dict:
        """Delegate validated inference to the engine."""
        return self._engine.infer_validated(system_prompt, query, max_tokens)

    def unload(self) -> None:
        self._engine.unload()

    @property
    def backend_type(self) -> str:
        if not _gpu_available():
            return "cpu"
        return "gpu"


class OpenRouterBackend(InferenceBackend):
    """Wraps OpenRouterClient for cloud inference via OpenRouter API."""

    def __init__(self, client) -> None:
        """Wrap an OpenRouterClient instance.

        Args:
            client: os_agent.inference.openrouter.OpenRouterClient instance.
        """
        self._client = client

    @property
    def client(self):
        """Direct access to the underlying OpenRouterClient."""
        return self._client

    def infer(self, system_prompt: str, user_message: str,
              max_tokens: int | None = None) -> str:
        return self._client.infer(system_prompt, user_message, max_tokens)

    def infer_streaming(self, system_prompt: str, user_message: str,
                        max_tokens: int | None = None) -> Generator[str, None, None]:
        yield from self._client.infer_streaming(system_prompt, user_message, max_tokens)

    def unload(self) -> None:
        self._client.close()

    @property
    def backend_type(self) -> str:
        return "openrouter"


class BackendManager:
    """Manages the active inference backend and model switching.

    Thread-safe: switch operations are serialised with a lock so concurrent
    D-Bus SwitchModel calls don't race.
    """

    def __init__(self, config: dict, registry,
                 initial_backend: InferenceBackend | None,
                 initial_model_name: str) -> None:
        """
        Args:
            config: daemon.yaml config dict (model + generation sections).
            registry: ModelRegistry instance for model lookup/validation.
            initial_backend: The backend loaded at daemon startup, or None
                if the daemon started in lazy-load (not_loaded) state.
            initial_model_name: Name of the initially loaded model, or "" if
                not_loaded.
        """
        self._config = config
        self._registry = registry
        self._active: InferenceBackend | None = initial_backend
        self._active_model_name = initial_model_name
        self._lock = threading.Lock()

    @property
    def active(self) -> InferenceBackend | None:
        """Return the currently active backend (or None if not loaded)."""
        return self._active

    @property
    def active_model_name(self) -> str:
        """Name of the currently loaded model, or '' if not loaded."""
        return self._active_model_name

    def update_config(self, config: dict) -> None:
        """Replace the stored config. Used when settings.yaml is reloaded."""
        self._config = config

    def _resolve_local_model(self, model_name: str) -> tuple[bool, str, str]:
        """Validate a local model and return (ok, model_path, error)."""
        model_info = self._registry.get_model(model_name)
        if model_info is None:
            return False, "", f"Model '{model_name}' not found in registry"
        if model_info.type != "gguf":
            return False, "", f"Model '{model_name}' is not a local GGUF model"
        model_path = model_info.path
        if not Path(model_path).exists():
            return False, "", f"Model file not found: {model_path}"
        if not self._registry.validate_gguf(model_path):
            return False, "", f"Invalid GGUF file: {model_path}"
        return True, model_path, ""

    def _build_engine_config(self, model_path: str) -> dict:
        """Shallow-merge a fresh model path into the stored config."""
        new_config = dict(self._config)
        new_model_cfg = dict(new_config.get("model", {}))
        new_model_cfg["path"] = model_path
        new_config["model"] = new_model_cfg
        return new_config

    def load_local(self, model_name: str) -> tuple[bool, str]:
        """Load a local GGUF model from the not_loaded state (or replace
        the current backend if one is already active).

        Unlike switch_to_local, this tolerates `_active is None` so it is
        the entry point for the lazy-load fast-path.

        Returns (success, error_message).
        """
        with self._lock:
            ok, model_path, err = self._resolve_local_model(model_name)
            if not ok:
                return False, err

            from os_agent.inference.engine import InferenceEngine
            try:
                new_engine = InferenceEngine(self._build_engine_config(model_path))
            except Exception as e:
                return False, f"Failed to load model: {e}"

            # New engine loaded — now unload old if any
            if self._active is not None:
                try:
                    self._active.unload()
                except Exception as e:
                    log.warning("Unload of previous backend failed: %s", e)

            self._active = LocalBackend(new_engine)
            self._active_model_name = model_name
            log.info("Loaded local model: %s", model_name)
            return True, ""

    def unload_current(self) -> tuple[bool, str]:
        """Unload whatever is active and transition to not_loaded.

        Safe to call when nothing is loaded — returns success in that case.
        Returns (success, error_message).
        """
        with self._lock:
            if self._active is None:
                return True, ""
            try:
                self._active.unload()
            except Exception as e:
                log.error("Unload failed: %s", e)
                return False, str(e)
            self._active = None
            self._active_model_name = ""
            log.info("Backend unloaded — daemon now in not_loaded state")
            return True, ""

    def apply_generation_params(self, params: dict) -> dict:
        """Hot-apply generation params to the active backend.

        No-op (returns {}) for OpenRouter (params are sent per-call) and
        for the not_loaded state.
        """
        if self._active is None:
            return {}
        if not isinstance(self._active, LocalBackend):
            return {}
        engine = self._active.engine
        if engine is None or not getattr(engine, "loaded", False):
            return {}
        return engine.update_generation_params(params)

    def switch_to_local(self, model_name: str) -> tuple[bool, str]:
        """Switch to a local GGUF model.

        Thin wrapper over load_local — kept for API compatibility with Step 6
        callers (dbus_service._switch_to_local).
        """
        return self.load_local(model_name)

    def switch_to_openrouter(self, api_key: str, model_id: str) -> tuple[bool, str]:
        """Switch to OpenRouter cloud inference.

        Tests the connection first. If test fails, keeps current backend
        and returns failure.

        Returns (success, error_message).
        """
        with self._lock:
            from os_agent.inference.openrouter import OpenRouterClient

            gen_cfg = self._config.get("generation", {})
            client = OpenRouterClient(
                api_key=api_key,
                model_id=model_id,
                temperature=gen_cfg.get("temperature", 0.3),
                max_tokens=gen_cfg.get("max_tokens", 1024),
            )

            # Test connection before unloading local model
            success, error = client.test_connection()
            if not success:
                log.warning("OpenRouter connection test failed: %s", error)
                return False, error

            # Test passed — unload local backend (if any) and switch
            if self._active is not None:
                try:
                    self._active.unload()
                except Exception as e:
                    log.warning("Unload of previous backend failed: %s", e)
            self._active = OpenRouterBackend(client)
            self._active_model_name = f"openrouter:{model_id}"
            log.info("Switched to OpenRouter model: %s", model_id)
            return True, ""

    def list_models(self) -> list[dict]:
        """Return all available models with active flag set.

        Returns list of dicts with: name, path, type, description, active.
        """
        registered = self._registry.list_models()
        discovered = self._registry.scan_model_dir()

        result = []
        for model in registered + discovered:
            result.append({
                "name": model.name,
                "path": model.path,
                "type": model.type,
                "description": model.description,
                "active": model.name == self._active_model_name,
            })

        # Add OpenRouter entry if configured
        or_config = self._registry.get_openrouter_config()
        active_type = self._active.backend_type if self._active is not None else ""
        if or_config.get("enabled") or active_type == "openrouter":
            or_model = or_config.get("default_model", "deepseek/deepseek-chat")
            or_name = f"openrouter:{or_model}"
            result.append({
                "name": or_name,
                "path": "https://openrouter.ai/api/v1",
                "type": "openrouter",
                "description": f"Cloud inference via OpenRouter ({or_model})",
                "active": active_type == "openrouter",
            })

        return result


def _gpu_available() -> bool:
    """Check if an NVIDIA GPU is available via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
