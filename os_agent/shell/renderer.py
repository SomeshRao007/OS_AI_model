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
        "risk-safe": "#98c379",
        "risk-moderate": "#e5c07b bold",
        "risk-dangerous": "#e06c75 bold",
    }
)

def _escape(text: str) -> str:
    """Escape text for prompt_toolkit HTML output."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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
                '<title>?</title><info> prefix for chatbot. </info>'
                '<title>/ai</title><info> for co-pilot mode.</info>'
            ),
            style=NEUROSH_STYLE,
        )
        print()

    def print_help(self) -> None:
        lines = [
            "<title>neurosh commands:</title>",
            "",
            "  <title>/chatbot</title>   Switch to chatbot mode",
            "  <title>/terminal</title>  Switch to terminal (bash) mode",
            "  <title>/ai</title>        Switch to AI co-pilot mode",
            "  <title>/history</title>   Show command history",
            "  <title>/memory</title>    Show agent memory stats",
            "  <title>/agents</title>    Show active agents",
            "  <title>/clear</title>     Clear session context",
            "  <title>/help</title>      Show this help",
            "  <title>/exit</title>      Exit neurosh",
            "",
            "  <title>?</title> &lt;query&gt;   Chatbot query from terminal mode",
            "  <title>!</title> &lt;cmd&gt;     Bash command from chatbot mode",
        ]
        for line in lines:
            print_formatted_text(HTML(line), style=NEUROSH_STYLE)

    def print_risk_badge(self, risk: str, command: str) -> None:
        """Display a risk-classified command with color-coded badge."""
        tag = f"risk-{risk}"
        label = risk.upper()
        if risk == "dangerous":
            label = "WARNING: DANGEROUS"
        print_formatted_text(
            HTML(f"<{tag}>[{label}]</{tag}> Execute: {_escape(command)}"),
            style=NEUROSH_STYLE,
        )

    def print_execution_output(
        self, stdout: str, stderr: str, exit_code: int, timed_out: bool
    ) -> None:
        """Display command execution results."""
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        if stderr:
            self.print_error(stderr.rstrip())
        if timed_out:
            self.print_error("(timed out)")
        elif exit_code != 0:
            self.print_error(f"(exit code: {exit_code})")

    def print_meta_response(self, title: str, body: str) -> None:
        print_formatted_text(HTML(f"<title>{title}</title>"), style=NEUROSH_STYLE)
        if body:
            print(body)
