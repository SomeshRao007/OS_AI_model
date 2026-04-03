"""Entry point: python -m os_agent"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="os_agent",
        description="Agentic OS — AI-powered system intelligence daemon",
    )
    parser.add_argument(
        "--daemon", action="store_true", help="Run as background daemon (systemd mode)"
    )
    parser.add_argument(
        "--shell", action="store_true", help="Launch neurosh interactive shell"
    )
    parser.add_argument(
        "--version", action="version", version="os_agent 0.1.0"
    )
    args = parser.parse_args()

    if args.daemon:
        from os_agent.ipc.dbus_service import run_daemon
        run_daemon()
        return

    # Default: launch neurosh shell
    from os_agent.shell import NeuroshShell

    shell = NeuroshShell()
    shell.run()


if __name__ == "__main__":
    main()
