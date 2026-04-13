"""KWallet profile helper — single secure store for OpenRouter credentials.

All OpenRouter API keys live in KWallet under a single entry, holding a
JSON document with multiple named profiles and a ``current`` pointer:

    {
      "profiles": {
        "personal": {"key": "sk-or-...", "last_model": "anthropic/claude-sonnet-4-6"},
        "work":     {"key": "sk-or-...", "last_model": "openai/gpt-4o"}
      },
      "current": "personal"
    }

Why a single blob instead of one-entry-per-profile:
    - atomic read/write (one KWallet D-Bus call, no partial state)
    - avoids relying on folder-listing output parsing
    - the JSON shape is trivially versionable

KWallet access uses D-Bus directly (org.kde.kwalletd6) rather than
shelling out to ``kwallet-query``.  The KF5 ``kwallet-query`` binary
cannot talk to the KF6 ``kwalletd6`` service that Plasma 6 runs, so
the subprocess approach fails on any Plasma 6 system.

All functions are defensive: KWallet missing, wallet locked, entry
absent, malformed JSON — each returns an empty/default value and logs
at DEBUG. **Nothing in this module raises out to callers**, because the
daemon needs to keep running even if the wallet is unavailable (it just
falls back to the local model with a notification).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import dbus

log = logging.getLogger("ai-daemon.kwallet")

_APP_ID = "ai-daemon"
_FOLDER = "AI-OS"
_ENTRY = "openrouter_profiles"

# D-Bus coordinates for KWallet 6 (Plasma 6 / KDE Frameworks 6)
_KW_SERVICE = "org.kde.kwalletd6"
_KW_PATH = "/modules/kwalletd6"
_KW_IFACE = "org.kde.KWallet"

# Empty document shape — returned when KWallet has no stored profiles
# yet, or when the stored JSON is malformed. Callers can safely index
# into ``profiles`` / ``current``.
_EMPTY_DOC: dict[str, Any] = {"profiles": {}, "current": ""}


# ── D-Bus helpers ─────────────────────────────────────────────────────────

def _get_iface():
    """Return the org.kde.KWallet D-Bus interface proxy, or None."""
    try:
        bus = dbus.SessionBus()
        proxy = bus.get_object(_KW_SERVICE, _KW_PATH)
        return dbus.Interface(proxy, _KW_IFACE)
    except dbus.exceptions.DBusException as exc:
        log.debug("Cannot connect to kwalletd6: %s", exc)
        return None


def _open_wallet(iface) -> int:
    """Open the local wallet and return a handle (>= 0), or -1 on failure."""
    try:
        wallet_name = str(iface.localWallet())
        handle = int(iface.open(wallet_name, dbus.Int64(0), _APP_ID))
        if handle < 0:
            log.debug("kwalletd6 refused to open wallet %r (handle=%d)",
                       wallet_name, handle)
        return handle
    except dbus.exceptions.DBusException as exc:
        log.debug("Cannot open wallet: %s", exc)
        return -1


# ── Public API ────────────────────────────────────────────────────────────

def is_available() -> bool:
    """Quick health check — is kwalletd6 reachable?"""
    iface = _get_iface()
    if iface is None:
        return False
    try:
        iface.localWallet()
        return True
    except dbus.exceptions.DBusException:
        return False


def load_profiles() -> dict[str, Any]:
    """Return the full profiles document, or an empty skeleton.

    Safe to call unconditionally — callers do not need to check
    ``is_available`` first.
    """
    iface = _get_iface()
    if iface is None:
        return dict(_EMPTY_DOC)

    handle = _open_wallet(iface)
    if handle < 0:
        return dict(_EMPTY_DOC)

    try:
        if not bool(iface.hasFolder(handle, _FOLDER, _APP_ID)):
            return dict(_EMPTY_DOC)

        raw = str(iface.readPassword(handle, _FOLDER, _ENTRY, _APP_ID))
        if not raw:
            return dict(_EMPTY_DOC)

        data = json.loads(raw)

        if not isinstance(data, dict):
            log.warning("kwallet entry %s is not an object — resetting", _ENTRY)
            return dict(_EMPTY_DOC)

        # Normalise shape so downstream code never has to defensive-check
        data.setdefault("profiles", {})
        data.setdefault("current", "")
        if not isinstance(data["profiles"], dict):
            data["profiles"] = {}
        if not isinstance(data["current"], str):
            data["current"] = ""
        return data

    except json.JSONDecodeError as exc:
        log.warning("kwallet entry %s has malformed JSON: %s", _ENTRY, exc)
        return dict(_EMPTY_DOC)
    except dbus.exceptions.DBusException as exc:
        log.debug("kwallet read failed: %s", exc)
        return dict(_EMPTY_DOC)
    finally:
        try:
            iface.close(handle, False, _APP_ID)
        except dbus.exceptions.DBusException:
            pass


def save_profiles(doc: dict[str, Any]) -> bool:
    """Write the full profiles document back to KWallet.

    Returns True on success, False otherwise (caller can decide whether
    to surface the error to the user). Does not raise.
    """
    iface = _get_iface()
    if iface is None:
        log.error("save_profiles: kwalletd6 not reachable")
        return False

    handle = _open_wallet(iface)
    if handle < 0:
        log.error("save_profiles: cannot open wallet")
        return False

    try:
        # Create the folder on first use
        if not bool(iface.hasFolder(handle, _FOLDER, _APP_ID)):
            ok = bool(iface.createFolder(handle, _FOLDER, _APP_ID))
            if not ok:
                log.error("save_profiles: cannot create folder %r", _FOLDER)
                return False

        payload = json.dumps(doc, separators=(",", ":"))
        rc = int(iface.writePassword(
            handle, _FOLDER, _ENTRY, payload, _APP_ID))
        if rc != 0:
            log.error("save_profiles: writePassword returned %d", rc)
            return False
        return True

    except dbus.exceptions.DBusException as exc:
        log.error("save_profiles: D-Bus error: %s", exc)
        return False
    finally:
        try:
            iface.close(handle, False, _APP_ID)
        except dbus.exceptions.DBusException:
            pass


# ── Profile CRUD ───────────────────────────────────────────────────────────

def list_profile_names() -> list[str]:
    """Return all profile names sorted, or ``[]`` if none / KWallet missing."""
    doc = load_profiles()
    return sorted(doc.get("profiles", {}).keys())


def list_profiles_masked() -> list[dict[str, Any]]:
    """Return profile metadata with **masked** keys for D-Bus transport.

    Shape:
        [{"name": str, "last_model": str, "masked_key": str, "is_current": bool}, ...]

    Plaintext keys never leave this helper unless ``reveal_key`` is called.
    """
    doc = load_profiles()
    current = doc.get("current", "")
    result: list[dict[str, Any]] = []
    for name, profile in sorted(doc.get("profiles", {}).items()):
        key = profile.get("key", "") if isinstance(profile, dict) else ""
        result.append({
            "name": name,
            "last_model": profile.get("last_model", "") if isinstance(profile, dict) else "",
            "masked_key": _mask_key(key),
            "is_current": (name == current),
        })
    return result


def get_profile(name: str) -> tuple[str, str] | None:
    """Return ``(key, last_model)`` for the named profile, or None.

    The key is **plaintext** — callers should only use this for
    constructing an OpenRouterClient in the daemon, never log it.
    """
    doc = load_profiles()
    profile = doc.get("profiles", {}).get(name)
    if not isinstance(profile, dict):
        return None
    key = profile.get("key", "")
    if not key:
        return None
    return key, profile.get("last_model", "")


def get_current_profile() -> tuple[str, str, str] | None:
    """Return ``(name, key, last_model)`` for the active profile, or None.

    Returns None if: no profiles exist, ``current`` is empty, or the
    current pointer references a missing profile.
    """
    doc = load_profiles()
    name = doc.get("current", "")
    if not name:
        return None
    got = get_profile(name)
    if got is None:
        return None
    key, model = got
    return name, key, model


def reveal_key(name: str) -> str:
    """Return the plaintext key for a profile (or empty string if missing).

    Used only when the UI's eye-toggle explicitly asks to view a key.
    """
    got = get_profile(name)
    return got[0] if got else ""


def upsert_profile(name: str, key: str, last_model: str) -> bool:
    """Add or replace a profile. Does not touch ``current``.

    Returns True on success. Rejects empty name or empty key.
    """
    if not name or not key:
        log.warning("upsert_profile: refusing empty name or key")
        return False
    doc = load_profiles()
    doc.setdefault("profiles", {})[name] = {
        "key": key,
        "last_model": last_model or "",
    }
    return save_profiles(doc)


def delete_profile(name: str) -> bool:
    """Remove a profile. If it was current, clear ``current``.

    Never auto-picks a replacement — the user must explicitly re-select.
    """
    if not name:
        return False
    doc = load_profiles()
    profiles = doc.get("profiles", {})
    if name not in profiles:
        return True  # already gone — treat as success
    del profiles[name]
    if doc.get("current") == name:
        doc["current"] = ""
    return save_profiles(doc)


def set_current_profile(name: str) -> bool:
    """Set the active profile pointer. Empty string clears it.

    Rejects names that don't exist in the store (so the UI can't leave
    ``current`` pointing at a ghost).
    """
    doc = load_profiles()
    if name and name not in doc.get("profiles", {}):
        log.warning("set_current_profile: no such profile %r", name)
        return False
    doc["current"] = name or ""
    return save_profiles(doc)


def update_last_model(name: str, model_id: str) -> bool:
    """Remember the most recently used OpenRouter model for a profile.

    Called by the daemon whenever a switch to ``openrouter:<id>`` with
    this profile succeeds, so the next session resumes on the same model.
    """
    if not name:
        return False
    doc = load_profiles()
    profile = doc.get("profiles", {}).get(name)
    if not isinstance(profile, dict):
        return False
    profile["last_model"] = model_id
    return save_profiles(doc)


# ── Internal helpers ───────────────────────────────────────────────────────

def _mask_key(key: str) -> str:
    """Return a display-safe masked form of an API key.

    Shows the provider prefix (``sk-or-``) if present and the last four
    characters, with the middle replaced by bullets. Empty input returns
    empty string so the UI can show "(no key)".
    """
    if not key:
        return ""
    if len(key) <= 8:
        return "\u2022" * len(key)
    prefix = ""
    body = key
    for pfx in ("sk-or-v1-", "sk-or-", "sk-"):
        if key.startswith(pfx):
            prefix = pfx
            body = key[len(pfx):]
            break
    tail = body[-4:] if len(body) >= 4 else body
    return f"{prefix}{'\u2022' * 8}{tail}"
