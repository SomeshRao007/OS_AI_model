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
    classify_git_risk,
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
        self._timeout = config.get("timeout_seconds", 30)
        self._bwrap_path = shutil.which("bwrap")
        self._enabled = config.get("enabled", True) and self._bwrap_works()

    def _bwrap_works(self) -> bool:
        """Test if bwrap can actually create sandboxes in this environment.

        Environments like VS Code terminals or restrictive AppArmor policies
        can block user namespace creation even when bwrap is installed.
        """
        if not self._bwrap_path:
            _log.warning("bwrap not installed — running commands without sandbox")
            return False

        probe = subprocess.run(
            [self._bwrap_path, "--bind", "/", "/", "--dev", "/dev",
             "--proc", "/proc", "--die-with-parent",
             "--", "/bin/true"],
            capture_output=True, timeout=5,
        )
        if probe.returncode != 0:
            stderr = probe.stderr.decode(errors="replace").strip()
            _log.warning("bwrap sandbox unavailable (%s) — running commands directly", stderr)
            return False
        return True

    def classify_risk(self, command: str) -> str:
        """Classify a command as safe, moderate, or dangerous.

        Order: dangerous patterns first (regex on full string),
        then git subcommand check, then safe set, else moderate.
        """
        for pattern, _reason in DANGEROUS_PATTERNS:
            if pattern.search(command):
                return RiskLevel.DANGEROUS

        base_cmds = extract_base_commands(command)

        # Git commands: safety depends on subcommand (status=safe, push=moderate)
        if base_cmds and base_cmds[0] == "git":
            return classify_git_risk(command)

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
