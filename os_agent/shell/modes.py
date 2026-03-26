"""Mode switching logic for neurosh shell.

Two modes: terminal (commands → bash) and chatbot (queries → master agent).
Handles ?, !, and / prefix routing.
"""

from __future__ import annotations


class ShellMode:
    TERMINAL = "terminal"
    CHATBOT = "chatbot"


class ModeManager:
    """Manages shell mode state and classifies user input by prefix."""

    def __init__(
        self,
        default_mode: str = ShellMode.TERMINAL,
        chatbot_prefix: str = "?",
        terminal_prefix: str = "!",
    ):
        self._mode = default_mode
        self._chatbot_prefix = chatbot_prefix
        self._terminal_prefix = terminal_prefix

    @property
    def mode(self) -> str:
        return self._mode

    def switch_to_chatbot(self) -> None:
        self._mode = ShellMode.CHATBOT

    def switch_to_terminal(self) -> None:
        self._mode = ShellMode.TERMINAL

    def classify_input(self, raw: str) -> tuple[str, str]:
        """Determine effective mode and clean the input.

        Returns:
            (effective_mode, cleaned_input) where effective_mode is
            "terminal", "chatbot", or "meta".
        """
        if raw.startswith("/"):
            return "meta", raw

        # ? prefix forces chatbot from terminal mode
        if raw.startswith(self._chatbot_prefix) and self._mode == ShellMode.TERMINAL:
            return ShellMode.CHATBOT, raw[len(self._chatbot_prefix) :].lstrip()

        # ! prefix forces terminal from chatbot mode
        if raw.startswith(self._terminal_prefix) and self._mode == ShellMode.CHATBOT:
            return ShellMode.TERMINAL, raw[len(self._terminal_prefix) :].lstrip()

        return self._mode, raw

    def prompt_text(self) -> str:
        """Plain-text prompt string for the current mode."""
        if self._mode == ShellMode.CHATBOT:
            return "neurosh[chatbot]> "
        return "neurosh> "
