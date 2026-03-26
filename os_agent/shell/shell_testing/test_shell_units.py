"""Unit tests for neurosh shell components (no model required)."""

from os_agent.shell.modes import ModeManager, ShellMode
from os_agent.shell.history import ShellHistory
from os_agent.shell.completer import create_completer
from os_agent.shell.renderer import Renderer


def test_mode_manager():
    """Test mode switching and input classification."""
    mgr = ModeManager()

    # Default mode is terminal
    assert mgr.mode == ShellMode.TERMINAL
    assert mgr.prompt_text() == "neurosh> "

    # Normal input in terminal mode → terminal
    assert mgr.classify_input("ls -la") == ("terminal", "ls -la")

    # ? prefix in terminal mode → chatbot
    assert mgr.classify_input("? find large files") == ("chatbot", "find large files")

    # / prefix → meta regardless of mode
    assert mgr.classify_input("/help") == ("meta", "/help")
    assert mgr.classify_input("/chatbot") == ("meta", "/chatbot")

    # Switch to chatbot mode
    mgr.switch_to_chatbot()
    assert mgr.mode == ShellMode.CHATBOT
    assert mgr.prompt_text() == "neurosh[chatbot]> "

    # Normal input in chatbot mode → chatbot
    assert mgr.classify_input("find stuff") == ("chatbot", "find stuff")

    # ! prefix in chatbot mode → terminal
    assert mgr.classify_input("!uname -a") == ("terminal", "uname -a")

    # ? prefix in chatbot mode → stays chatbot (no special meaning)
    assert mgr.classify_input("? test") == ("chatbot", "? test")

    # ! prefix in terminal mode → stays terminal (no special meaning)
    mgr.switch_to_terminal()
    assert mgr.classify_input("!echo hi") == ("terminal", "!echo hi")

    print("PASS: test_mode_manager")


def test_shell_history():
    """Test annotated history tracking."""
    hist = ShellHistory(max_entries=5)

    hist.add_terminal("ls", 0)
    hist.add_terminal("false", 1)
    hist.add_chatbot("find large files", "files")

    entries = hist.recent()
    assert len(entries) == 3
    assert entries[0].mode == "terminal"
    assert entries[0].exit_code == 0
    assert entries[2].mode == "chatbot"
    assert entries[2].domain == "files"

    # Test max entries cap
    for i in range(10):
        hist.add_terminal(f"cmd_{i}", 0)
    assert len(hist.recent(100)) == 5  # capped at max_entries

    # Test format_display
    output = hist.format_display()
    assert "[TERMINAL]" in output
    assert "cmd_" in output

    print("PASS: test_shell_history")


def test_completer():
    """Test that completers are created for both modes."""
    terminal = create_completer("terminal")
    chatbot = create_completer("chatbot")
    assert terminal is not None
    assert chatbot is not None
    print("PASS: test_completer")


def test_renderer():
    """Test renderer instantiation (output tests require a terminal)."""
    r = Renderer()
    r.print_info("test info")
    r.print_error("test error")
    r.print_success("test success")
    r.print_domain_badge("files")
    r.print_domain_badge("network")
    r.print_help()
    r.print_meta_response("Title", "body text")
    print("PASS: test_renderer")


def test_imports():
    """Verify full import chain works."""
    from os_agent.shell import NeuroshShell
    from os_agent.agents.base import BaseAgent
    from os_agent.agents.master import MasterAgent

    assert hasattr(BaseAgent, "augmented_prompt")
    assert hasattr(MasterAgent, "get_agent")
    print("PASS: test_imports")


if __name__ == "__main__":
    test_mode_manager()
    test_shell_history()
    test_completer()
    test_renderer()
    test_imports()
    print("\n=== ALL UNIT TESTS PASSED ===")
