"""Environment context for AI mode — CWD, directory contents, system info.

Gathers lightweight contextual information to inject into agent prompts
so the AI co-pilot is aware of the user's actual environment.
"""

from __future__ import annotations

import os
import platform
import subprocess


class EnvironmentContext:
    """Gathers and formats environment information for prompt injection.

    system_info() is cached once at session start (~40 tokens).
    cwd_context() is dynamic, recalculated per query (~30 tokens).
    """

    def __init__(self) -> None:
        self._system_info: str | None = None

    def system_info(self) -> str:
        """One-time system info string, cached after first call."""
        if self._system_info is not None:
            return self._system_info
        self._system_info = self._gather_system_info()
        return self._system_info

    def cwd_context(self) -> str:
        """Current working directory + brief contents summary."""
        cwd = os.getcwd()
        entries = sorted(os.listdir(cwd))
        shown = entries[:10]
        summary = ", ".join(shown)
        if len(entries) > 10:
            summary += f" ... (+{len(entries) - 10} more)"
        return f"CWD: {cwd}\nContents: {summary}"

    def full_context(self) -> str:
        """Combined system info + CWD context for prompt injection."""
        return f"{self.system_info()}\n{self.cwd_context()}"

    def _gather_system_info(self) -> str:
        uname = platform.uname()
        username = os.getenv("USER", "unknown")
        distro = self._get_distro()
        ram = self._get_total_ram()
        gpu = self._get_gpu()

        parts = [
            f"OS: {uname.system} {uname.release}",
            f"Host: {uname.node}",
            f"User: {username}",
            f"Distro: {distro}",
            f"Python: {platform.python_version()}",
        ]
        if ram:
            parts.append(f"RAM: {ram}")
        if gpu:
            parts.append(f"GPU: {gpu}")

        return " | ".join(parts)

    @staticmethod
    def _get_distro() -> str:
        """Read distro name from os-release."""
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        return line.split("=", 1)[1].strip().strip('"')
        except OSError:
            pass
        return "unknown"

    @staticmethod
    def _get_total_ram() -> str:
        """Read total RAM from /proc/meminfo."""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        gb = kb / (1024 * 1024)
                        return f"{gb:.0f}GB"
        except (OSError, ValueError):
            pass
        return ""

    @staticmethod
    def _get_gpu() -> str:
        """Get GPU model from nvidia-smi (single line)."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return ""
