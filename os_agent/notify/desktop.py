"""Desktop notifications via notify-send.

Notifications are alerts only — the actual Accept/Reject confirmation stays
in the terminal (prompt_toolkit's input loop is incompatible with the GLib
event loop that D-Bus action callbacks require).

Gracefully degrades if notify-send is not installed.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

_log = logging.getLogger("neurosh.notify")


class DesktopNotifier:
    """Fire-and-forget desktop notifications via notify-send."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self._enabled = cfg.get("enabled", True)
        self._notify_on_moderate = cfg.get("notify_on_moderate", False)
        self._vram_threshold_mb = cfg.get("vram_threshold_mb", 500)
        self._bin = shutil.which("notify-send")

        if self._enabled and not self._bin:
            _log.warning(
                "notify-send not found — desktop notifications disabled. "
                "Install with: sudo apt install libnotify-bin"
            )
            self._enabled = False

    @property
    def available(self) -> bool:
        return self._enabled and self._bin is not None

    @property
    def vram_threshold_mb(self) -> int:
        return self._vram_threshold_mb

    def notify(
        self,
        title: str,
        body: str,
        urgency: str = "normal",
        icon: str = "dialog-information",
        expire_ms: int = 10000,
    ) -> None:
        """Send a desktop notification. Non-blocking, fire-and-forget."""
        if not self.available:
            return

        cmd = [
            self._bin,
            f"--urgency={urgency}",
            f"--icon={icon}",
            f"--expire-time={expire_ms}",
            "--app-name=neurosh",
            title,
            body,
        ]

        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)

    def warn_dangerous_command(self, command: str, risk_level: str, domain: str) -> None:
        """Alert the user that a dangerous/moderate command needs confirmation."""
        if risk_level == "moderate" and not self._notify_on_moderate:
            return

        truncated = command[:120]
        urgency = "critical" if risk_level == "dangerous" else "normal"
        icon = "dialog-warning" if risk_level == "dangerous" else "dialog-information"

        self.notify(
            title=f"neurosh [{risk_level.upper()}]",
            body=f"{truncated}\n\nDomain: {domain} — confirm in terminal",
            urgency=urgency,
            icon=icon,
        )

    def warn_vram_low(self, free_mb: int, total_mb: int) -> None:
        """Alert when free VRAM drops below threshold."""
        self.notify(
            title="neurosh — GPU VRAM Low",
            body=(
                f"Free VRAM: {free_mb} MB / {total_mb} MB\n"
                f"Threshold: {self._vram_threshold_mb} MB\n"
                "Inference may fail or slow down."
            ),
            urgency="critical",
            icon="dialog-warning",
            expire_ms=0,  # persistent until dismissed
        )


def check_vram_and_warn(engine, notifier: DesktopNotifier) -> None:
    """Check VRAM before inference and fire a notification if low.

    Called once per inference, not per token. nvidia-smi call takes ~50ms.
    """
    if not notifier.available:
        return

    vram = engine.get_vram_usage()
    free = vram.get("free", 0)
    total = vram.get("total", 0)

    if free > 0 and free < notifier.vram_threshold_mb:
        notifier.warn_vram_low(free, total)
