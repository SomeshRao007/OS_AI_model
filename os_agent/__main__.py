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
        print("daemon mode not implemented yet (Step 9)")
        sys.exit(1)

    # Default: launch neurosh shell
    print("neurosh shell not implemented yet (Step 6)")
    print("Use: python os_agent/test_inference.py --quick  (to verify inference)")
    sys.exit(0)


if __name__ == "__main__":
    main()
