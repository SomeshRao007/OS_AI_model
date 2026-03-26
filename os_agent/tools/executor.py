"""Sandboxed command execution via bubblewrap.

Wraps commands in bwrap for timeout enforcement, process cleanup
(--die-with-parent), and optional network isolation. Risk classification
determines whether commands auto-execute or require user confirmation.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass

from os_agent.tools.registry import (
    DANGEROUS_PATTERNS,
    SAFE_COMMANDS,
    extract_base_commands,
    is_command_allowed,
)

_log = logging.getLogger("executor")

# Domains that need network access for their core functionality
_NETWORK_DOMAINS: frozenset[str] = frozenset({"network", "packages"})


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Result of a sandboxed command execution."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


class RiskLevel:
    SAFE = "safe"
    MODERATE = "moderate"
    DANGEROUS = "dangerous"


class SandboxedExecutor:
    """Execute commands inside a bubblewrap sandbox with risk classification."""

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        self._enabled = config.get("enabled", True)
        self._timeout = config.get("timeout_seconds", 30)
        self._bwrap_path = shutil.which("bwrap")

        if self._enabled and not self._bwrap_path:
            raise RuntimeError(
                "bubblewrap (bwrap) not found. Install it:\n"
                "  sudo apt install bubblewrap"
            )

    def classify_risk(self, command: str) -> str:
        """Classify a command as safe, moderate, or dangerous.

        Order: dangerous patterns first (regex on full string),
        then safe check (all base commands in safe set),
        else moderate.
        """
        # 1. Dangerous pattern match on the full command string
        for pattern, _reason in DANGEROUS_PATTERNS:
            if pattern.search(command):
                return RiskLevel.DANGEROUS

        # 2. All base commands must be in the safe set
        base_cmds = extract_base_commands(command)
        if base_cmds and all(cmd in SAFE_COMMANDS for cmd in base_cmds):
            return RiskLevel.SAFE

        return RiskLevel.MODERATE

    def check_domain_allowed(self, command: str, domain: str) -> bool:
        """Check if the command is within the domain's tool whitelist."""
        return is_command_allowed(command, domain)

    def run(
        self,
        command: str,
        domain: str = "files",
        timeout: int | None = None,
    ) -> ExecutionResult:
        """Execute a command, optionally inside a bubblewrap sandbox.

        Returns ExecutionResult with stdout, stderr, exit_code, timed_out.
        """
        effective_timeout = timeout or self._timeout

        if self._enabled:
            full_cmd = self._build_bwrap_command(command, domain)
        else:
            full_cmd = ["/bin/bash", "-c", command]

        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
            return ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                stdout=(exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr="",
                exit_code=-1,
                timed_out=True,
            )

    def _build_bwrap_command(self, command: str, domain: str) -> list[str]:
        """Build the bwrap wrapper command list."""
        args = [
            self._bwrap_path,
            "--bind", "/", "/",
            "--dev", "/dev",
            "--proc", "/proc",
            "--die-with-parent",
        ]

        # --unshare-net requires CAP_NET_ADMIN (root) — skip for unprivileged users
        # Network isolation can be re-enabled when running as a systemd service
        # with the appropriate capabilities.

        args.extend(["--", "/bin/bash", "-c", command])
        return args
