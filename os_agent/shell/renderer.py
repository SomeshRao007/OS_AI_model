"""Styled terminal output for neurosh shell.

Uses prompt_toolkit HTML formatting for colored, structured output.
"""

from __future__ import annotations

from prompt_toolkit import print_formatted_text, HTML
from prompt_toolkit.styles import Style

NEUROSH_STYLE = Style.from_dict(
    {
        "domain-files": "#e5c07b bold",
        "domain-network": "#61afef bold",
        "domain-process": "#c678dd bold",
        "domain-packages": "#98c379 bold",
        "domain-kernel": "#e06c75 bold",
        "info": "ansigray",
        "error": "#e06c75",
        "success": "#98c379",
        "title": "bold",
    }
)

_DOMAIN_TAGS = {
    "files": "domain-files",
    "network": "domain-network",
    "process": "domain-process",
    "packages": "domain-packages",
    "kernel": "domain-kernel",
}


class Renderer:
    """All styled terminal output for neurosh."""

    def print_domain_badge(self, domain: str) -> None:
        tag = _DOMAIN_TAGS.get(domain, "info")
        print_formatted_text(
            HTML(f"<{tag}>[{domain.upper()}]</{tag}>"), style=NEUROSH_STYLE
        )

    def print_info(self, msg: str) -> None:
        print_formatted_text(HTML(f"<info>{msg}</info>"), style=NEUROSH_STYLE)

    def print_error(self, msg: str) -> None:
        print_formatted_text(HTML(f"<error>{msg}</error>"), style=NEUROSH_STYLE)

    def print_success(self, msg: str) -> None:
        print_formatted_text(HTML(f"<success>{msg}</success>"), style=NEUROSH_STYLE)

    def print_welcome(self, vram: dict[str, int] | None = None) -> None:
        print_formatted_text(
            HTML("<title>neurosh</title> <info>v0.1.0 — Neural OS Shell</info>"),
            style=NEUROSH_STYLE,
        )
        if vram:
            print_formatted_text(
                HTML(
                    f"<info>GPU: {vram.get('used', 0)} MB used / "
                    f"{vram.get('total', 0)} MB total</info>"
                ),
                style=NEUROSH_STYLE,
            )
        print_formatted_text(
            HTML(
                '<info>Type </info><title>/help</title>'
                '<info> for commands. </info>'
                '<title>?</title><info> prefix for AI queries.</info>'
            ),
            style=NEUROSH_STYLE,
        )
        print()

    def print_help(self) -> None:
        lines = [
            "<title>neurosh commands:</title>",
            "",
            "  <title>/ai</title>        Switch to AI mode",
            "  <title>/direct</title>    Switch to direct (bash) mode",
            "  <title>/history</title>   Show command history",
            "  <title>/memory</title>    Show agent memory stats",
            "  <title>/agents</title>    Show active agents",
            "  <title>/clear</title>     Clear session context",
            "  <title>/help</title>      Show this help",
            "  <title>/exit</title>      Exit neurosh",
            "",
            "  <title>?</title> &lt;query&gt;   AI query from direct mode",
            "  <title>!</title> &lt;cmd&gt;     Bash command from AI mode",
        ]
        for line in lines:
            print_formatted_text(HTML(line), style=NEUROSH_STYLE)

    def print_meta_response(self, title: str, body: str) -> None:
        print_formatted_text(HTML(f"<title>{title}</title>"), style=NEUROSH_STYLE)
        if body:
            print(body)
