"""User settings loader for the AI OS daemon.

Reads/writes ``~/.config/ai-daemon/settings.yaml`` — the user-facing
configuration owned by the Settings KCM. This is the mutable layer that
overlays the read-only ``daemon.yaml`` defaults baked into the ISO.

Single writer: the KCM writes directly, then calls
``org.aios.Daemon.ReloadSettings()`` to make the daemon re-read the file.
This avoids a file-watcher race and lets the user edit settings while the
daemon is stopped.
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("ai-daemon.settings")

SETTINGS_DIR = Path(
    os.environ.get("AI_DAEMON_SETTINGS_DIR",
                   os.path.expanduser("~/.config/ai-daemon"))
)
SETTINGS_PATH = SETTINGS_DIR / "settings.yaml"

# Defaults for every key the Settings UI can touch. Anything not listed
# here is treated as an advanced/daemon-side option and stays in daemon.yaml.
#
# The ``active`` section is the persistence layer for "which backend was
# the user on last?" — the daemon reads this on startup to resume the
# user's choice across reboots / systemctl restarts. Writing it is the
# responsibility of ``SwitchModel`` on the D-Bus service.
#
# ``openrouter`` intentionally has no hardcoded model ID. The per-user
# model preference lives inside the KWallet profile (see os_agent.kwallet)
# so we can support multiple profiles each with their own last-used model.
DEFAULT_SETTINGS: dict[str, Any] = {
    "active": {
        "backend": "local",            # "local" | "openrouter"
        "local_model": "",             # model name in the local registry
        "openrouter_profile": "",      # name of the KWallet profile to use
    },
    "model": {
        "default_local": "qwen3.5-4b-os-q4km",
        "lazy_load": True,
        "auto_start": False,
    },
    "generation": {
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "max_tokens": 1024,
        "n_ctx": 2048,          # reload-required
        "n_gpu_layers": -1,     # reload-required
        "seed": -1,
    },
    "openrouter": {
        "enabled": False,
    },
    "behaviour": {
        "notify_on_moderate": False,
        "vram_threshold_mb": 500,
    },
}


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Return a new dict with overlay merged on top of base (recursive)."""
    out = copy.deepcopy(base)
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_settings() -> dict:
    """Return merged settings: defaults overlaid with user settings.yaml.

    Missing or malformed file → return defaults (never raises).
    """
    if not SETTINGS_PATH.exists():
        return copy.deepcopy(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        if not isinstance(user, dict):
            log.warning("settings.yaml is not a mapping — ignoring")
            return copy.deepcopy(DEFAULT_SETTINGS)
        return _deep_merge(DEFAULT_SETTINGS, user)
    except (OSError, yaml.YAMLError) as e:
        log.warning("Failed to load settings.yaml: %s — using defaults", e)
        return copy.deepcopy(DEFAULT_SETTINGS)


def save_settings(settings: dict) -> None:
    """Atomically write the given settings dict to settings.yaml (mode 0644).

    Writes to a sibling tmp file then os.replace to avoid torn reads.
    Creates the parent directory if needed.
    """
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_PATH.with_suffix(".yaml.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(settings, f, default_flow_style=False, sort_keys=False)
    os.chmod(tmp, 0o644)
    os.replace(tmp, SETTINGS_PATH)


def update_active(backend: str, local_model: str = "",
                  openrouter_profile: str = "") -> dict:
    """Persist the "which backend is currently active" state.

    Loads the current settings, merges the ``active`` section in place,
    writes atomically, and returns the merged dict. Called after a
    successful SwitchModel so a daemon restart picks the same backend
    back up.

    Args:
        backend: "local" or "openrouter".
        local_model: Model name in the local registry (for backend=local).
        openrouter_profile: KWallet profile name (for backend=openrouter).
    """
    current = load_settings()
    current.setdefault("active", {})
    current["active"]["backend"] = backend
    if backend == "local":
        current["active"]["local_model"] = local_model
        # Leave openrouter_profile alone — user may still have it set
        # and expect to flip back to cloud from the UI.
    else:
        current["active"]["openrouter_profile"] = openrouter_profile
    save_settings(current)
    return current


def apply_to_daemon_config(daemon_config: dict, settings: dict) -> dict:
    """Overlay user settings onto a daemon.yaml dict for engine consumption.

    Returns a new dict. Used on daemon startup and on ReloadSettings().
    Only the keys the KCM exposes are copied; advanced daemon.yaml keys
    (shell, memory, sandbox, agents) are left untouched.
    """
    out = copy.deepcopy(daemon_config)
    out.setdefault("model", {})
    out.setdefault("generation", {})

    gen = settings.get("generation", {})
    for key in ("temperature", "top_p", "top_k", "repeat_penalty",
                "max_tokens", "n_ctx", "n_gpu_layers", "seed"):
        if key in gen:
            out["generation"][key] = gen[key]

    model = settings.get("model", {})
    if "n_gpu_layers" in gen:
        # n_gpu_layers also belongs under model: for InferenceEngine
        out["model"]["n_gpu_layers"] = gen["n_gpu_layers"]
    if "n_ctx" in gen:
        out["model"]["n_ctx"] = gen["n_ctx"]
    return out
