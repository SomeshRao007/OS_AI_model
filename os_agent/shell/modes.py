"""Mode switching logic for neurosh shell.

Two modes: direct (commands → bash) and AI (queries → master agent).
Handles ?, !, and / prefix routing.
"""

from __future__ import annotations


class ShellMode:
    DIRECT = "direct"
    AI = "ai"


class ModeManager:
    """Manages shell mode state and classifies user input by prefix."""

    def __init__(
        self,
        default_mode: str = ShellMode.DIRECT,
        ai_prefix: str = "?",
        direct_prefix: str = "!",
    ):
        self._mode = default_mode
        self._ai_prefix = ai_prefix
        self._direct_prefix = direct_prefix

    @property
    def mode(self) -> str:
        return self._mode

    def switch_to_ai(self) -> None:
        self._mode = ShellMode.AI

    def switch_to_direct(self) -> None:
        self._mode = ShellMode.DIRECT

    def classify_input(self, raw: str) -> tuple[str, str]:
        """Determine effective mode and clean the input.

        Returns:
            (effective_mode, cleaned_input) where effective_mode is
            "direct", "ai", or "meta".
        """
        if raw.startswith("/"):
            return "meta", raw

        # ? prefix forces AI from direct mode
        if raw.startswith(self._ai_prefix) and self._mode == ShellMode.DIRECT:
            return ShellMode.AI, raw[len(self._ai_prefix) :].lstrip()

        # ! prefix forces direct from AI mode
        if raw.startswith(self._direct_prefix) and self._mode == ShellMode.AI:
            return ShellMode.DIRECT, raw[len(self._direct_prefix) :].lstrip()

        return self._mode, raw

    def prompt_text(self) -> str:
        """Plain-text prompt string for the current mode."""
        if self._mode == ShellMode.AI:
            return "neurosh[ai]> "
        return "neurosh> "
