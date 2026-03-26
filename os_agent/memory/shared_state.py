"""Tier 1: Global shared state — JSON-backed, all agents R/W.

Stores system snapshots (cached), agent action history, and cross-agent
context so specialists can see what other agents have done.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


class SharedState:
    """Global state persisted to a JSON file with atomic writes."""

    _CACHE_TTL = 30.0  # seconds before system snapshot refreshes
    _MAX_ACTION_LOG = 50

    def __init__(self, state_dir: str) -> None:
        self._state_dir = Path(state_dir).expanduser()
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self._state_dir / "state.json"
        self._state = self._load()
        self._snapshot_cache: dict | None = None
        self._snapshot_time: float = 0.0

    # --- public API ---

    def get(self, key: str, default=None):
        return self._state.get(key, default)

    def set(self, key: str, value) -> None:
        self._state[key] = value
        self._save()

    def snapshot(self) -> dict:
        """Return cached system snapshot (disk, network, processes)."""
        now = time.time()
        if self._snapshot_cache and (now - self._snapshot_time) < self._CACHE_TTL:
            return self._snapshot_cache

        self._snapshot_cache = {
            "disk_usage": self._run_cmd("df -h --output=target,pcent,avail | head -15"),
            "ip_addresses": self._run_cmd("ip -brief addr"),
            "top_processes": self._run_cmd("ps aux --sort=-%mem | head -10"),
        }
        self._snapshot_time = now
        return self._snapshot_cache

    def log_action(self, domain: str, query: str, summary: str) -> None:
        """Append to action history (capped at _MAX_ACTION_LOG entries)."""
        log = self._state.get("action_log", [])
        log.append({
            "domain": domain,
            "query": query,
            "summary": summary,
            "timestamp": time.time(),
        })
        # keep only the most recent entries
        self._state["action_log"] = log[-self._MAX_ACTION_LOG:]
        self._save()

    def cross_agent_context(self) -> str:
        """Return cross-agent context string (what other agents have done)."""
        return self._state.get("cross_context", "")

    def set_cross_context(self, context: str) -> None:
        self._state["cross_context"] = context
        self._save()

    # --- persistence (external I/O — try-except is acceptable here) ---

    def _load(self) -> dict:
        if not self._state_path.exists():
            return {}
        try:
            return json.loads(self._state_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self) -> None:
        """Atomic write: write to .tmp then os.replace."""
        tmp_path = self._state_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(json.dumps(self._state, indent=2, default=str))
            os.replace(tmp_path, self._state_path)
        except OSError:
            pass

    @staticmethod
    def _run_cmd(cmd: str) -> str:
        """Run a shell command for system snapshot. Returns stdout or error string."""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            return "(timeout)"
        except OSError:
            return "(error)"


# --- built-in test ---

if __name__ == "__main__" and "--test" in sys.argv:
    import tempfile

    print("=== SharedState Test ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        ss = SharedState(tmpdir)
        ss.set("test_key", "test_value")

        ss2 = SharedState(tmpdir)
        got = ss2.get("test_key")
        status = "PASS" if got == "test_value" else "FAIL"
        print(f"  Persistence: {status} (got={got!r})")

        ss2.log_action("files", "find large files", "found 5GB in /tmp")
        log = ss2.get("action_log", [])
        status = "PASS" if len(log) == 1 and log[0]["domain"] == "files" else "FAIL"
        print(f"  Action log:  {status} (entries={len(log)})")

        snap = ss2.snapshot()
        status = "PASS" if "disk_usage" in snap and snap["disk_usage"] else "FAIL"
        print(f"  Snapshot:    {status} (keys={list(snap.keys())})")

        ss2.set_cross_context("Files agent found 5GB in /tmp")
        ctx = ss2.cross_agent_context()
        status = "PASS" if "5GB" in ctx else "FAIL"
        print(f"  Cross ctx:   {status}")

        for i in range(60):
            ss2.log_action("test", f"query_{i}", f"result_{i}")
        log = ss2.get("action_log", [])
        status = "PASS" if len(log) == 50 else "FAIL"
        print(f"  Log cap:     {status} (count={len(log)})")

    print("=== Done ===")
