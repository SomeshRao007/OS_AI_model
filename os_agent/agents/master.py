"""Master agent — classifies queries and routes to domain specialists.

Two-stage classification:
  Stage 1: Keyword matching (instant, no GPU cost)
  Stage 2: Model-based (for ambiguous queries, ~50ms with max_tokens=10)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

from os_agent.agents.base import AgentResponse, BaseAgent
from os_agent.agents.files import FilesAgent
from os_agent.agents.kernel import KernelAgent
from os_agent.agents.network import NetworkAgent
from os_agent.agents.packages import PackagesAgent
from os_agent.agents.process import ProcessAgent
from os_agent.inference.engine import InferenceEngine
from os_agent.inference.prompt import MASTER_CLASSIFY_PROMPT
from os_agent.memory.agent_memory import AgentMemory
from os_agent.memory.session import SessionContext
from os_agent.memory.shared_state import SharedState

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "os_agent" / "config" / "daemon.yaml"

# ── Keyword sets per domain ──────────────────────────────────────────────
# Each set contains words that strongly signal a domain.
# Scored by set intersection with tokenized query words.

_DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "files": {
        "find", "chmod", "chown", "chgrp", "ln", "link", "symlink", "symbolic",
        "file", "files", "directory", "directories", "folder", "folders",
        "permission", "permissions", "ownership", "owner",
        "cp", "mv", "rm", "mkdir", "rmdir", "touch", "stat",
        "awk", "sed", "sort", "uniq", "wc", "grep", "head", "tail",
        "lines", "words", "characters", "column", "duplicate", "duplicates",
        "tar", "gzip", "gunzip", "zip", "unzip", "archive", "compress", "compressed",
        "mount", "umount", "usb", "filesystem", "filesystems",
        "bash", "script", "loop", "variable", "arguments", "argument",
    },
    "network": {
        "ssh", "scp", "rsync", "port", "ports", "tcp", "udp",
        "firewall", "iptables", "nftables", "dns", "ping", "traceroute",
        "network", "networking", "remote", "server", "curl", "wget",
        "telnet", "netstat", "ss", "ip",
    },
    "process": {
        "process", "processes", "pid", "kill", "killall", "pkill",
        "background", "foreground", "nohup", "disown", "job", "jobs",
        "cron", "crontab", "schedule", "scheduled",
        "cpu", "memory", "ram", "top", "htop", "ps",
        "systemctl", "service", "daemon",
        "user", "users", "useradd", "usermod", "userdel", "passwd",
        "group", "groups", "sudo", "lock", "account",
        "du", "df", "usage", "disk",
    },
    "packages": {
        "install", "uninstall", "package", "packages", "apt", "dpkg",
        "snap", "flatpak", "pip", "npm", "deb", "repository",
        "upgrade", "update", "owns", "debian", "ubuntu",
    },
    "kernel": {
        "kernel", "module", "modules", "modprobe", "lsmod", "modinfo",
        "proc", "thread", "threads", "paging", "virtual",
        "uname", "dmesg", "sysctl", "interrupt", "syscall",
    },
}

_VALID_DOMAINS = frozenset(_DOMAIN_KEYWORDS.keys())

# Split on non-alphanumeric chars to tokenize queries
_WORD_SPLIT = re.compile(r"[^a-z0-9]+")


class MasterAgent:
    """Routes user queries to the appropriate domain specialist.

    Owns the shared InferenceEngine, all specialist agents, and the
    three-tier memory system (shared state, per-domain FAISS, session).
    """

    def __init__(self, engine: InferenceEngine, config: dict | None = None):
        self._engine = engine

        if config is None:
            config = self._load_config()
        mem_cfg = config.get("memory", {})
        state_dir = mem_cfg.get("state_dir", "~/.local/share/ai-daemon")
        faiss_dims = mem_cfg.get("faiss_dims", 384)
        max_vectors = mem_cfg.get("max_vectors_per_domain", 500)

        self._shared_state = SharedState(state_dir)
        self._session = SessionContext()

        def _mem(domain: str) -> AgentMemory:
            return AgentMemory(domain, state_dir, faiss_dims, max_vectors)

        self._agents: dict[str, BaseAgent] = {
            "files": FilesAgent(memory=_mem("files")),
            "network": NetworkAgent(memory=_mem("network")),
            "process": ProcessAgent(memory=_mem("process")),
            "packages": PackagesAgent(memory=_mem("packages")),
            "kernel": KernelAgent(memory=_mem("kernel")),
        }

    def get_agent(self, domain: str) -> BaseAgent:
        """Return the specialist agent for the given domain."""
        return self._agents[domain]

    @property
    def shared_state(self) -> SharedState:
        return self._shared_state

    @property
    def session(self) -> SessionContext:
        return self._session

    def classify(self, query: str) -> str:
        """Return the domain name for a query. Keywords first, model fallback."""
        domain = self._classify_by_keywords(query)
        if domain:
            return domain
        return self._classify_by_model(query)

    def route(self, query: str) -> AgentResponse:
        """Classify, delegate, and update memory tiers."""
        domain = self.classify(query)
        agent = self._agents[domain]
        result = agent.handle(query, self._engine)

        self._session.add_turn(
            query, domain, result.response, result.memory_hits,
        )
        self._shared_state.log_action(domain, query, result.response[:100])

        return result

    @staticmethod
    def _load_config() -> dict:
        if _CONFIG_PATH.exists():
            return yaml.safe_load(_CONFIG_PATH.read_text()) or {}
        return {}

    def _classify_by_keywords(self, query: str) -> str | None:
        """Stage 1: fast keyword matching. Returns domain or None if ambiguous."""
        words = set(_WORD_SPLIT.split(query.lower()))

        scores: dict[str, int] = {}
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            hit_count = len(words & keywords)
            if hit_count > 0:
                scores[domain] = hit_count

        if not scores:
            return None

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Clear winner: top score strictly greater than second place
        if len(ranked) == 1:
            return ranked[0][0]
        if ranked[0][1] > ranked[1][1]:
            return ranked[0][0]

        # Tie — fall through to model
        return None

    def _classify_by_model(self, query: str) -> str:
        """Stage 2: use the LLM when keywords are ambiguous.

        max_tokens=512 because the model generates a ~200-300 token <think>
        block before the domain word. _strip_thinking() handles removal.
        """
        raw = self._engine.infer(MASTER_CLASSIFY_PROMPT, query, max_tokens=512)
        domain = raw.strip().lower().split()[0] if raw.strip() else "files"

        if domain not in _VALID_DOMAINS:
            return "files"

        return domain


# ── Test infrastructure ──────────────────────────────────────────────────
# Expected routing for our 5-domain architecture (150 questions)
# Lazy-loaded from eval_questions.py — not imported at module level so
# the shell can start without the test data file present.
_TEST_CASES: list[tuple[str, str]] | None = None


def _get_test_cases() -> list[tuple[str, str]]:
    global _TEST_CASES
    if _TEST_CASES is None:
        from os_agent.Agent_benchmark_testing.eval_questions import ROUTING_TEST_CASES
        _TEST_CASES = ROUTING_TEST_CASES
    return _TEST_CASES


def _run_test_keywords():
    """Test 1: keyword classifier only, no model needed."""
    print("=" * 70)
    print("TEST 1: Keyword classifier only (no model)")
    print("=" * 70)

    master_keywords_only = MasterAgent.__new__(MasterAgent)
    master_keywords_only._agents = {}

    resolved = 0
    needs_model = []

    for question, expected in _get_test_cases():
        words = set(_WORD_SPLIT.split(question.lower()))
        scores = {}
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            hit_count = len(words & keywords)
            if hit_count > 0:
                scores[domain] = hit_count

        if not scores:
            result = None
        else:
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
                result = ranked[0][0]
            else:
                result = None

        if result is None:
            needs_model.append((question, expected, scores))
            print(f"  MODEL  {expected:10s}  scores={scores}  <- {question[:55]}")
        elif result == expected:
            resolved += 1
            print(f"  OK     {expected:10s}  ({result}, score={scores.get(result, 0)})  <- {question[:55]}")
        else:
            resolved += 1
            marker = "WRONG" if result != expected else "OK"
            print(f"  {marker:5s}  exp={expected:10s} got={result:10s}  scores={scores}  <- {question[:55]}")

    total = len(_get_test_cases())
    print(f"\nKeyword resolved: {resolved}/{total} ({resolved/total*100:.1f}%)")
    print(f"Needs model fallback: {len(needs_model)}/{total}")
    if needs_model:
        print("\nQuestions needing model:")
        for q, exp, scores in needs_model:
            print(f"  [{exp}] {q[:65]}  scores={scores}")


def _run_test_routing(engine: InferenceEngine):
    """Test 2: full routing (keywords + model fallback)."""
    print("=" * 70)
    print("TEST 2: Full routing (keywords + model fallback)")
    print("=" * 70)

    master = MasterAgent(engine)
    correct = 0
    keyword_resolved = 0
    model_resolved = 0
    failures = []

    for question, expected in _get_test_cases():
        # Check if keywords would resolve it
        kw_result = master._classify_by_keywords(question)
        predicted = master.classify(question)
        method = "keyword" if kw_result is not None else "model"

        if method == "keyword":
            keyword_resolved += 1
        else:
            model_resolved += 1

        if predicted == expected:
            correct += 1
            print(f"  OK    [{method:7s}] {expected:10s} <- {question[:55]}")
        else:
            failures.append((question, expected, predicted, method))
            print(f"  FAIL  [{method:7s}] exp={expected:10s} got={predicted:10s} <- {question[:55]}")

    total = len(_get_test_cases())
    pct = correct / total * 100
    print(f"\nRouting accuracy: {correct}/{total} ({pct:.1f}%)")
    print(f"  Keyword resolved: {keyword_resolved}")
    print(f"  Model resolved:   {model_resolved}")
    print(f"  Target: >= 90% ({int(total * 0.9)}/{total})")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for q, exp, got, method in failures:
            print(f"  [{method}] Expected {exp}, got {got}: {q[:65]}")

    return pct >= 90


def _run_test_model(engine: InferenceEngine):
    """Test 3: force model classification on all 44 questions."""
    print("=" * 70)
    print("TEST 3: Model classifier only (bypass keywords)")
    print("=" * 70)

    master = MasterAgent(engine)
    correct = 0
    failures = []

    for question, expected in _get_test_cases():
        raw = engine.infer(MASTER_CLASSIFY_PROMPT, question, max_tokens=512)
        parsed = raw.strip().lower().split()[0] if raw.strip() else "(empty)"

        if parsed == expected:
            correct += 1
            print(f"  OK    {expected:10s} raw={raw.strip()!r:15s} <- {question[:50]}")
        else:
            failures.append((question, expected, parsed, raw.strip()))
            print(f"  FAIL  exp={expected:10s} got={parsed:10s} raw={raw.strip()!r:15s} <- {question[:50]}")

    total = len(_get_test_cases())
    pct = correct / total * 100
    print(f"\nModel classification accuracy: {correct}/{total} ({pct:.1f}%)")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for q, exp, got, raw in failures:
            print(f"  Expected {exp}, got {got} (raw: {raw!r}): {q[:60]}")


def _run_test_e2e(engine: InferenceEngine):
    """Test 4: end-to-end routing + response (5 representative questions)."""
    print("=" * 70)
    print("TEST 4: End-to-end routing + response")
    print("=" * 70)

    master = MasterAgent(engine)

    # One question per domain
    e2e_questions = [
        ("Find all files larger than 100MB on Linux", "files"),
        ("List all open TCP ports on the system", "network"),
        ("How do I kill a process by name without knowing its PID?", "process"),
        ("How do I install a .deb package manually?", "packages"),
        ("What is a Linux kernel module and how do you load one?", "kernel"),
    ]

    for question, expected in e2e_questions:
        result = master.route(question)
        status = "OK" if result.domain == expected else "FAIL"
        print(f"\n  [{status}] [{result.domain.upper()}] {question}")
        print(f"  {'-' * 50}")
        print(f"  {result.response}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m os_agent.agents.master <test-mode>")
        print("  --test-keywords  : Test keyword classifier only (no model needed)")
        print("  --test-routing   : Test full routing (keywords + model)")
        print("  --test-model     : Test model classifier on all 44 questions")
        print("  --test-e2e       : Test end-to-end routing + response (5 questions)")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "--test-keywords":
        _run_test_keywords()
    elif mode in ("--test-routing", "--test-model", "--test-e2e"):
        print("Loading inference engine...")
        engine = InferenceEngine()
        print("Engine loaded.\n")

        if mode == "--test-routing":
            passed = _run_test_routing(engine)
            sys.exit(0 if passed else 1)
        elif mode == "--test-model":
            _run_test_model(engine)
        elif mode == "--test-e2e":
            _run_test_e2e(engine)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
