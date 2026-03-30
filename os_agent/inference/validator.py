"""Rule-based command validator: parse generated bash commands and catch
type mismatches, hallucinated flags, and deprecated syntax before execution.

Validation rules (checked in order):
  1. Explicitly invalid flag (e.g. wc -n)
  2. Unknown flag — not in help_db schema → block
  3. Deprecated syntax (find -perm +NNN) → block + suggest
  4. Arg type mismatch (flag expects filepath, got email) → block + suggest
  5. Duplicate flag → block
  6. No-arg command used with flags (nohup -d) → block (covered by rule 2)
  7. Distro mismatch (dnf/pacman on Debian) → warn only

Unknown commands (not in help_db) → soft warn, never block.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_HELP_DB_PATH = _PROJECT_ROOT / "os_agent" / "config" / "help_db.json"

_help_db: dict | None = None

# Non-Debian package managers — warn if seen on a Debian/Ubuntu system.
_REDHAT_PKG_MANAGERS = frozenset({"dnf", "yum", "rpm", "zypper", "pacman", "emerge", "xbps-install"})

# Commands with hand-curated, COMPLETE flag schemas (from OVERRIDES in build_help_db.py).
# Unknown-flag blocking (rule 2) only runs for these. Auto-parsed --help schemas are
# incomplete so blocking unknown flags there causes false positives on valid commands.
_CURATED_COMMANDS = frozenset({
    "ssh-keygen", "nohup", "find", "chage", "useradd", "wc", "gdb", "perf",
})


def _load_db() -> dict:
    global _help_db
    if _help_db is None:
        if not _HELP_DB_PATH.exists():
            _help_db = {}
        else:
            with open(_HELP_DB_PATH, encoding="utf-8") as f:
                _help_db = json.load(f)
    return _help_db


def infer_arg_type(arg: str) -> str:
    """Classify an argument string: email | filepath | number | string.

    Filepath check runs before email so that paths like ~/.ssh/user@host.com
    (which the model often generates when it knows -f needs a filepath) are
    correctly classified as filepath rather than email.
    """
    if not arg:
        return "string"
    # Filepath first: must precede email check because model-generated paths
    # can contain @ (e.g. ~/.ssh/user@host.com) and are still filepaths.
    if arg.startswith(("/", "~", "./", "../")) or (
        "/" in arg and not arg.startswith("-")
    ):
        return "filepath"
    # Email: user@domain.tld (only reached if not a filepath)
    if re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", arg):
        return "email"
    # Number: integer with optional leading sign
    if re.match(r"^[+-]?\d+$", arg):
        return "number"
    return "string"


def _extract_first_command_tokens(bash_command: str) -> list[str]:
    """Tokenise the first command in a pipeline / compound statement.

    Strips trailing & (background), splits on | && || ; and returns shlex
    tokens for just the first segment. Falls back to whitespace split on
    shlex errors (e.g. unmatched quotes in complex commands).
    """
    cmd = bash_command.strip()
    # Remove trailing background operator
    if cmd.endswith(" &"):
        cmd = cmd[:-2].strip()
    elif cmd.endswith("&"):
        cmd = cmd[:-1].strip()

    # Split off pipeline / compound operators — take first segment only
    for sep in (" | ", " || ", " && ", "; ", ";"):
        idx = cmd.find(sep)
        if idx != -1:
            cmd = cmd[:idx].strip()

    try:
        return shlex.split(cmd)
    except ValueError:
        return cmd.split()


def _find_suggestion(flags_schema: dict, got_type: str) -> str | None:
    """Find a flag in the schema whose expected type is compatible with got_type.

    Used to suggest the correct flag when the user provides an email where a
    filepath is expected (e.g. ssh-keygen -f email → suggest -C).
    """
    # email given to filepath flag → look for a string flag (comments/labels)
    if got_type == "email":
        for flag, info in flags_schema.items():
            if info.get("expects") == "string":
                hint = info.get("hint", "")
                if any(w in hint.lower() for w in ("email", "comment", "label", "string")):
                    return flag
        # fallback: any string flag
        for flag, info in flags_schema.items():
            if info.get("expects") == "string":
                return flag
    return None


def validate(bash_command: str) -> dict:
    """Validate a bash command against the help_db flag schema.

    Returns one of:
      {"ok": True}
      {"ok": True,  "warn": "..."}          ← unknown command, soft warn
      {"ok": False, "error": "...", "suggestion": "..."}  ← blocked
    """
    if not bash_command or not bash_command.strip():
        return {"ok": True}

    tokens = _extract_first_command_tokens(bash_command)
    if not tokens:
        return {"ok": True}

    cmd = tokens[0]
    # Strip path prefix (e.g. /usr/bin/find → find)
    cmd_name = cmd.split("/")[-1]
    args = tokens[1:]

    # ── Rule 7: distro mismatch ──────────────────────────────────────────
    if cmd_name in _REDHAT_PKG_MANAGERS:
        return {
            "ok": True,
            "warn": (
                f"{cmd_name}: this is a non-Debian package manager. "
                "On Ubuntu/Debian use apt/dpkg instead."
            ),
        }

    db = _load_db()
    entry = db.get(cmd_name)

    # ── Unknown command ───────────────────────────────────────────────────
    if entry is None:
        return {
            "ok": True,
            "warn": f"{cmd_name}: not in help_db, skipping flag validation",
        }

    flags_schema: dict = entry.get("flags", {})
    invalid_flags: list = entry.get("invalid_flags", [])

    # Walk tokens, tracking (flag, arg_value) pairs.
    # For find's -exec: skip everything between -exec and \; or + (belongs to subcommand).
    seen_flags: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]

        # Skip -exec ... \; blocks — flags inside belong to the subcommand, not find
        if cmd_name == "find" and token in ("-exec", "-execdir", "-ok"):
            i += 1
            while i < len(args) and args[i] not in (";", "\\;", "+"):
                i += 1
            i += 1  # skip the terminator itself
            continue

        if not token.startswith("-"):
            i += 1
            continue

        # Handle --flag=value style
        if "=" in token and token.startswith("--"):
            flag, _, arg_val = token.partition("=")
        else:
            flag = token
            arg_val = None

        # ── Rule 1: explicitly invalid flag ─────────────────────────────
        if flag in invalid_flags:
            # Build a helpful suggestion from valid flags
            valid = ", ".join(
                f"{f} ({info.get('hint', info.get('expects', ''))})"
                for f, info in flags_schema.items()
                if info.get("hint") or info.get("expects")
            )
            suggestion = f"Valid flags: {valid}" if valid else ""
            return {
                "ok": False,
                "error": f"{cmd_name}: invalid flag {flag} (this flag does not exist)",
                "suggestion": suggestion,
            }

        # ── Rule 2: unknown flag (curated commands only) ─────────────────
        # Only block unknown flags for commands with complete hand-curated schemas.
        # Auto-parsed --help schemas are incomplete — enforcing them causes false
        # positives on valid flags that --help parsing simply missed.
        if cmd_name in _CURATED_COMMANDS and flag not in flags_schema:
            candidates = [
                f"{f} ({info.get('hint', '')})"
                for f, info in flags_schema.items()
                if info.get("hint")
            ]
            suggestion = (
                f"Known flags: {', '.join(candidates[:4])}"
                if candidates
                else entry.get("note", entry.get("summary", ""))
            )
            return {
                "ok": False,
                "error": f"{cmd_name}: unknown flag {flag}",
                "suggestion": suggestion,
            }

        flag_info = flags_schema.get(flag, {})
        expects = flag_info.get("expects") if flag_info else None

        # Determine the flag's argument value
        if arg_val is None and expects is not None:
            # Consume next token as argument if it doesn't look like a flag.
            # Allow negative numbers (-1, -30) — they start with - but are values
            # (e.g. find -mtime -1 means "modified in the last 24 hours").
            if i + 1 < len(args):
                next_tok = args[i + 1]
                is_negative_num = bool(re.match(r"^-\d+", next_tok))
                if not next_tok.startswith("-") or is_negative_num:
                    arg_val = next_tok
                    i += 1  # consume the arg token

        # ── Rule 3: deprecated syntax ────────────────────────────────────
        # find -perm +NNN (old BSD syntax, removed in GNU findutils 4.5.12)
        if cmd_name == "find" and flag == "-perm" and arg_val and arg_val.startswith("+"):
            fixed = "/" + arg_val[1:]
            return {
                "ok": False,
                "error": f"find: -perm +{arg_val[1:]} is deprecated syntax",
                "suggestion": f"Use -perm {fixed} instead (e.g. find / -perm /4000 -type f)",
            }

        # ── Rule 4: arg type mismatch ────────────────────────────────────
        # Shell command substitutions ($(...) or `...`) are runtime-evaluated —
        # we can't know their type statically, so skip type checking for them.
        # e.g. perf -p $(pgrep nginx) is valid even though -p expects a number.
        if arg_val and (arg_val.startswith("$(") or arg_val.startswith("`")):
            seen_flags.append(flag)
            i += 1
            continue

        if arg_val and expects and expects not in ("command", "size_spec", "permission_mode", "enum"):
            got_type = infer_arg_type(arg_val)
            if got_type != expects and not (expects == "string" and got_type in ("string", "number", "email")):
                suggestion = ""
                if got_type == "email" and expects == "filepath":
                    alt_flag = _find_suggestion(flags_schema, "email")
                    if alt_flag:
                        alt_hint = flags_schema[alt_flag].get("hint", "")
                        suggestion = (
                            f"Did you mean {alt_flag}? "
                            f"({alt_hint}) "
                            f"Use {flag} for a file path, {alt_flag} for email/comment."
                        )
                    else:
                        suggestion = f"{flag} expects a file path, not an email address."
                return {
                    "ok": False,
                    "error": (
                        f"{cmd_name} {flag} expects {expects}, "
                        f"got {got_type} ({arg_val!r})"
                    ),
                    "suggestion": suggestion,
                }

        # ── Rule 5: duplicate flag ───────────────────────────────────────
        if flag in seen_flags:
            return {
                "ok": False,
                "error": f"{cmd_name}: duplicate flag {flag}",
                "suggestion": f"Remove the duplicate {flag}.",
            }
        seen_flags.append(flag)

        i += 1

    return {"ok": True}
