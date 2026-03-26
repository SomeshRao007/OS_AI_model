"""Step 7 verification tests for registry, executor, parser, context, and modes."""

from __future__ import annotations

import sys
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

results: list[str] = []


def log(msg: str) -> None:
    print(msg)
    results.append(msg)


def test_registry() -> None:
    from os_agent.tools.registry import (
        DOMAIN_WHITELIST, SAFE_COMMANDS, DANGEROUS_PATTERNS,
        extract_base_commands, is_command_allowed,
    )

    assert len(DOMAIN_WHITELIST) == 5, f"Expected 5 domains, got {len(DOMAIN_WHITELIST)}"
    assert "ls" in SAFE_COMMANDS
    assert len(DANGEROUS_PATTERNS) > 10, "Expected 10+ dangerous patterns"

    # Base command extraction
    assert extract_base_commands("sudo ls -la | grep foo") == ["ls", "grep"]
    assert extract_base_commands("nohup python3 script.py &") == ["python3"]
    assert extract_base_commands("env VAR=1 nice ls") == ["ls"]
    assert extract_base_commands("cd /tmp && rm -rf old") == ["cd", "rm"]
    assert extract_base_commands("cat file || echo fallback") == ["cat", "echo"]

    # Domain whitelist checks
    assert is_command_allowed("ls -la", "files")
    assert not is_command_allowed("iptables -L", "files")
    assert is_command_allowed("iptables -L", "network")
    assert is_command_allowed("apt install vim", "packages")
    assert not is_command_allowed("apt install vim", "files")
    assert is_command_allowed("ls | grep foo", "files")
    assert is_command_allowed("systemctl status nginx", "process")
    assert not is_command_allowed("systemctl status nginx", "kernel")

    log("  registry .............. PASS")


def test_executor_risk() -> None:
    from os_agent.tools.executor import SandboxedExecutor, RiskLevel

    executor = SandboxedExecutor({"enabled": False, "timeout_seconds": 5})

    # Safe commands
    assert executor.classify_risk("ls -la") == RiskLevel.SAFE
    assert executor.classify_risk("df -h") == RiskLevel.SAFE
    assert executor.classify_risk("ps aux | grep python") == RiskLevel.SAFE
    assert executor.classify_risk("cat /etc/os-release") == RiskLevel.SAFE
    assert executor.classify_risk("du -sh /home") == RiskLevel.SAFE
    assert executor.classify_risk("ping -c 1 8.8.8.8") == RiskLevel.SAFE

    # Moderate commands
    assert executor.classify_risk("mkdir test") == RiskLevel.MODERATE
    assert executor.classify_risk("cp file1 file2") == RiskLevel.MODERATE
    assert executor.classify_risk("touch newfile") == RiskLevel.MODERATE
    assert executor.classify_risk("chmod 644 file") == RiskLevel.MODERATE
    assert executor.classify_risk("apt install vim") == RiskLevel.MODERATE

    # Dangerous commands
    assert executor.classify_risk("rm -rf /tmp/*") == RiskLevel.DANGEROUS
    assert executor.classify_risk("rm --force --recursive /") == RiskLevel.DANGEROUS
    assert executor.classify_risk("dd if=/dev/zero of=test") == RiskLevel.DANGEROUS
    assert executor.classify_risk("kill -9 1234") == RiskLevel.DANGEROUS
    assert executor.classify_risk("mkfs.ext4 /dev/sda1") == RiskLevel.DANGEROUS
    assert executor.classify_risk("find . -delete") == RiskLevel.DANGEROUS
    assert executor.classify_risk("chmod 777 /tmp") == RiskLevel.DANGEROUS
    assert executor.classify_risk("shutdown -h now") == RiskLevel.DANGEROUS
    assert executor.classify_risk("reboot") == RiskLevel.DANGEROUS
    assert executor.classify_risk("iptables -F") == RiskLevel.DANGEROUS

    # Pipeline with dangerous command inside
    assert executor.classify_risk("cat foo | rm -rf /") == RiskLevel.DANGEROUS

    log("  executor risk ......... PASS")


def test_executor_run() -> None:
    from os_agent.tools.executor import SandboxedExecutor

    executor = SandboxedExecutor({"enabled": False, "timeout_seconds": 5})

    # Basic execution
    result = executor.run("echo hello world", domain="files")
    assert result.stdout.strip() == "hello world"
    assert result.exit_code == 0
    assert not result.timed_out

    # Stderr
    result = executor.run("ls /nonexistent_path_12345", domain="files")
    assert result.exit_code != 0
    assert result.stderr

    # Timeout
    result = executor.run("sleep 10", domain="files", timeout=1)
    assert result.timed_out

    log("  executor run .......... PASS")


def test_executor_bwrap() -> None:
    bwrap = shutil.which("bwrap")
    if not bwrap:
        log("  executor bwrap ........ SKIP (bwrap not installed)")
        return

    from os_agent.tools.executor import SandboxedExecutor

    executor = SandboxedExecutor({"enabled": True, "timeout_seconds": 5})

    # Basic bwrap execution
    result = executor.run("echo bwrap works", domain="files")
    assert result.stdout.strip() == "bwrap works"
    assert result.exit_code == 0

    # Network isolation for files domain
    result = executor.run("ip addr show lo", domain="files")
    # Should work even with --unshare-net (loopback still exists)
    assert result.exit_code == 0

    log(f"  executor bwrap ........ PASS (found at {bwrap})")


def test_parser() -> None:
    from os_agent.tools.parser import extract_command, extract_all_commands

    # Standard bash code block
    resp1 = "Here is the command:\n```bash\ndu -sh .\n```\nThis shows disk usage."
    assert extract_command(resp1) == "du -sh ."

    # No code block
    resp2 = "Just run ls to see files."
    assert extract_command(resp2) is None

    # Multiple code blocks
    resp3 = "```bash\nfind . -size +100M\n```\nand also:\n```sh\ndf -h\n```"
    cmds = extract_all_commands(resp3)
    assert cmds == ["find . -size +100M", "df -h"]

    # Untagged code block
    resp4 = "Try this:\n```\nls -la /tmp\n```"
    assert extract_command(resp4) == "ls -la /tmp"

    # Multi-line command
    resp5 = "```bash\nfind / -name '*.log' \\\n  -size +10M\n```"
    cmd = extract_command(resp5)
    assert cmd is not None
    assert "find" in cmd

    # Empty code block
    resp6 = "```bash\n\n```"
    assert extract_command(resp6) is None

    log("  parser ................ PASS")


def test_context() -> None:
    from os_agent.shell.context import EnvironmentContext

    ctx = EnvironmentContext()
    si = ctx.system_info()
    assert "OS:" in si
    assert "User:" in si
    assert "Host:" in si

    cwd = ctx.cwd_context()
    assert "CWD:" in cwd
    assert "Contents:" in cwd

    full = ctx.full_context()
    assert "OS:" in full and "CWD:" in full

    # system_info should be cached (same object)
    si2 = ctx.system_info()
    assert si is si2

    log("  context ............... PASS")


def test_agent_response() -> None:
    from os_agent.agents.base import AgentResponse

    # Backward compatibility: old-style construction
    ar = AgentResponse(domain="files", response="test")
    assert ar.action_type is None
    assert ar.command is None

    # New fields
    ar2 = AgentResponse(
        domain="files", response="test", action_type="safe", command="ls -la"
    )
    assert ar2.action_type == "safe"
    assert ar2.command == "ls -la"

    log("  AgentResponse ......... PASS")


def test_modes() -> None:
    from os_agent.shell.modes import ShellMode, ModeManager

    assert ShellMode.AI == "ai"

    mgr = ModeManager()
    mgr.switch_to_ai()
    assert mgr.mode == "ai"

    # In AI mode, all non-meta input goes to AI handler
    mode, cleaned = mgr.classify_input("hello world")
    assert mode == "ai"
    assert cleaned == "hello world"

    mode2, cleaned2 = mgr.classify_input("/help")
    assert mode2 == "meta"

    # ? prefix is NOT special in AI mode
    mode3, cleaned3 = mgr.classify_input("? what is this")
    assert mode3 == "ai"
    assert cleaned3 == "? what is this"

    # Switch back to terminal
    mgr.switch_to_terminal()
    mode4, cleaned4 = mgr.classify_input("? disk space")
    assert mode4 == "chatbot"
    assert cleaned4 == "disk space"

    log("  modes ................. PASS")


def test_imports() -> None:
    """Verify all new/updated modules import cleanly."""
    from os_agent.tools import (
        ExecutionResult, RiskLevel, SandboxedExecutor,
        extract_command, extract_all_commands,
        DOMAIN_WHITELIST, SAFE_COMMANDS, is_command_allowed,
    )
    from os_agent.shell.context import EnvironmentContext
    from os_agent.shell.modes import ShellMode
    from os_agent.agents.base import AgentResponse, BaseAgent

    log("  imports ............... PASS")


def main() -> None:
    log("=" * 60)
    log("Step 7 Verification Tests")
    log("=" * 60)
    log("")

    tests = [
        test_imports,
        test_registry,
        test_executor_risk,
        test_executor_run,
        test_executor_bwrap,
        test_parser,
        test_context,
        test_agent_response,
        test_modes,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as exc:
            failed += 1
            log(f"  {test.__name__:.<25s} FAIL: {exc}")

    log("")
    log(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    log("=" * 60)

    # Save results
    results_path = Path(__file__).parent / "RESULTS.md"
    results_path.write_text("\n".join(results) + "\n")
    log(f"\nResults saved to {results_path}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
