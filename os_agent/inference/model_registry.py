"""Model registry — loads models.yaml and manages the catalog of available models.

Provides model lookup, GGUF validation, and discovery of unregistered models
dropped into the models directory.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

log = logging.getLogger("ai-daemon.registry")

# GGUF magic number: bytes 0x47 0x47 0x55 0x46 ("GGUF" in ASCII)
_GGUF_MAGIC = b"GGUF"

# Config discovery order for models.yaml
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Metadata for a single model entry."""

    name: str
    path: str  # filesystem path for local, model ID for openrouter
    type: str  # "gguf" or "openrouter"
    description: str = ""


class ModelRegistry:
    """Loads models.yaml and provides model lookup + validation.

    Config discovery order:
      1. Directory of AI_DAEMON_CONFIG env var (production: /opt/ai-daemon/config/)
      2. os_agent/config/ in the project root (development)
    """

    def __init__(self, config_dir: str | Path | None = None) -> None:
        if config_dir is not None:
            self._config_dir = Path(config_dir)
        else:
            self._config_dir = self._discover_config_dir()

        self._models_path = self._config_dir / "models.yaml"
        self._raw: dict = {}
        self._reload()

    def _reload(self) -> None:
        """(Re)load models.yaml from disk."""
        if not self._models_path.exists():
            log.warning("models.yaml not found at %s", self._models_path)
            self._raw = {}
            return

        with open(self._models_path) as f:
            parsed = yaml.safe_load(f) or {}
        if not isinstance(parsed, dict):
            log.error("models.yaml is not a valid YAML dict (got %s), ignoring",
                      type(parsed).__name__)
            self._raw = {}
            return
        self._raw = parsed
        log.info("Loaded %d local model(s) from %s",
                 len(self._raw.get("models", {}).get("local", [])),
                 self._models_path)

    def list_models(self) -> list[ModelInfo]:
        """Return all registered local models."""
        models_section = self._raw.get("models", {})
        result: list[ModelInfo] = []
        for entry in models_section.get("local", []):
            result.append(ModelInfo(
                name=entry.get("name", ""),
                path=entry.get("path", ""),
                type=entry.get("type", "gguf"),
                description=entry.get("description", ""),
            ))
        return result

    def get_model(self, name: str) -> ModelInfo | None:
        """Look up a model by name. Returns None if not found."""
        for model in self.list_models():
            if model.name == name:
                return model
        return None

    def get_default_model_name(self) -> str:
        """Return the name of the default model from models.yaml."""
        return self._raw.get("models", {}).get("default", "")

    def get_openrouter_config(self) -> dict:
        """Return the OpenRouter configuration section.

        Returns dict with keys: enabled, api_key, default_model.
        Empty dict if not configured.
        """
        return self._raw.get("models", {}).get("openrouter", {})

    def get_models_dir(self) -> Path:
        """Return the directory where local GGUF models are stored."""
        # Production: /opt/ai-daemon/models/
        # Development: inferred from first model's path, or fallback
        models = self.list_models()
        if models and models[0].path:
            return Path(models[0].path).parent
        return self._config_dir.parent / "models"

    def scan_model_dir(self) -> list[ModelInfo]:
        """Discover GGUF files in the models directory not already registered.

        Useful for finding models users dropped in without editing models.yaml.
        """
        models_dir = self.get_models_dir()
        if not models_dir.is_dir():
            return []

        registered_paths = {m.path for m in self.list_models()}
        discovered: list[ModelInfo] = []

        for gguf_file in sorted(models_dir.glob("*.gguf")):
            abs_path = str(gguf_file.resolve())
            if abs_path in registered_paths:
                continue
            if not self.validate_gguf(abs_path):
                log.warning("Skipping invalid GGUF: %s", abs_path)
                continue
            discovered.append(ModelInfo(
                name=gguf_file.stem,
                path=abs_path,
                type="gguf",
                description=f"Discovered model: {gguf_file.name}",
            ))

        return discovered

    @staticmethod
    def validate_gguf(path: str | Path) -> bool:
        """Check that a file exists and starts with the GGUF magic bytes."""
        p = Path(path)
        if not p.is_file():
            return False
        with open(p, "rb") as f:
            magic = f.read(4)
        return magic == _GGUF_MAGIC

    @staticmethod
    def _discover_config_dir() -> Path:
        """Find the config directory using env var or project structure."""
        env_config = os.environ.get("AI_DAEMON_CONFIG")
        if env_config:
            return Path(env_config).parent

        # Development fallback
        dev_path = _PROJECT_ROOT / "os_agent" / "config"
        if dev_path.is_dir():
            return dev_path

        # Production fallback
        return Path("/opt/ai-daemon/config")
