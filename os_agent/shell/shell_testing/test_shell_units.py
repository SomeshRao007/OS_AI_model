"""Unit tests for neurosh shell components (no model required)."""

from os_agent.shell.modes import ModeManager, ShellMode
from os_agent.shell.history import ShellHistory
from os_agent.shell.completer import create_completer
from os_agent.shell.renderer import Renderer


def test_mode_manager():
    """Test mode switching and input classification."""
    mgr = ModeManager()

    # Default mode is direct
    assert mgr.mode == ShellMode.DIRECT
    assert mgr.prompt_text() == "neurosh> "

    # Normal input in direct mode → direct
    assert mgr.classify_input("ls -la") == ("direct", "ls -la")

    # ? prefix in direct mode → AI
    assert mgr.classify_input("? find large files") == ("ai", "find large files")

    # / prefix → meta regardless of mode
    assert mgr.classify_input("/help") == ("meta", "/help")
    assert mgr.classify_input("/ai") == ("meta", "/ai")

    # Switch to AI mode
    mgr.switch_to_ai()
    assert mgr.mode == ShellMode.AI
    assert mgr.prompt_text() == "neurosh[ai]> "

    # Normal input in AI mode → AI
    assert mgr.classify_input("find stuff") == ("ai", "find stuff")

    # ! prefix in AI mode → direct
    assert mgr.classify_input("!uname -a") == ("direct", "uname -a")

    # ? prefix in AI mode → stays AI (no special meaning)
    assert mgr.classify_input("? test") == ("ai", "? test")

    # ! prefix in direct mode → stays direct (no special meaning)
    mgr.switch_to_direct()
    assert mgr.classify_input("!echo hi") == ("direct", "!echo hi")

    print("PASS: test_mode_manager")


def test_shell_history():
    """Test annotated history tracking."""
    hist = ShellHistory(max_entries=5)

    hist.add_direct("ls", 0)
    hist.add_direct("false", 1)
    hist.add_ai("find large files", "files")

    entries = hist.recent()
    assert len(entries) == 3
    assert entries[0].mode == "direct"
    assert entries[0].exit_code == 0
    assert entries[2].mode == "ai"
    assert entries[2].domain == "files"

    # Test max entries cap
    for i in range(10):
        hist.add_direct(f"cmd_{i}", 0)
    assert len(hist.recent(100)) == 5  # capped at max_entries

    # Test format_display
    output = hist.format_display()
    assert "[BASH]" in output
    assert "cmd_" in output

    print("PASS: test_shell_history")


def test_completer():
    """Test that completers are created for both modes."""
    direct = create_completer("direct")
    ai = create_completer("ai")
    assert direct is not None
    assert ai is not None
    print("PASS: test_completer")


def test_renderer():
    """Test renderer instantiation (output tests require a terminal)."""
    r = Renderer()
    # Just verify methods exist and don't crash with basic calls
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
