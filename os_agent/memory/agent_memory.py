"""Tier 2: Per-domain FAISS semantic memory.

Each specialist agent has its own AgentMemory instance with a separate FAISS
index. Solutions are stored with embeddings from all-MiniLM-L6-v2 (384-dim,
CPU-only) and persisted to disk as {domain}.faiss + {domain}.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class Solution:
    """A stored problem-solution pair from a domain agent."""
    query: str
    response: str
    domain: str
    timestamp: float


# L2^2 threshold for "similar enough" — corresponds to cosine similarity > 0.8
# for normalized embeddings: L2^2 = 2 * (1 - cos_sim), so 2*(1-0.8) = 0.4
_L2_SQ_THRESHOLD = 0.4


class AgentMemory:
    """FAISS-backed semantic memory for a single domain agent."""

    def __init__(
        self,
        domain: str,
        state_dir: str,
        faiss_dims: int = 384,
        max_vectors: int = 500,
    ) -> None:
        self._domain = domain
        self._faiss_dims = faiss_dims
        self._max_vectors = max_vectors
        self._memory_dir = Path(state_dir).expanduser() / "memory"
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._model = None  # lazy-loaded sentence-transformers
        self._index = None
        self._solutions: list[Solution] = []
        self._load_index()

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def count(self) -> int:
        return len(self._solutions)

    # --- public API ---

    def search(self, query: str, top_k: int = 3) -> list[Solution]:
        """Semantic search for similar past solutions. Filters by L2 distance."""
        if not self._index or self._index.ntotal == 0:
            return []

        vec = self._embed(query).reshape(1, -1)
        k = min(top_k, self._index.ntotal)
        distances_sq, indices = self._index.search(vec, k)

        results = []
        for dist_sq, idx in zip(distances_sq[0], indices[0]):
            if idx < 0 or idx >= len(self._solutions):
                continue
            if dist_sq > _L2_SQ_THRESHOLD:
                continue
            results.append(self._solutions[idx])
        return results

    def store(self, query: str, response: str) -> None:
        """Store a new solution. Auto-prunes if over capacity, then saves."""
        solution = Solution(
            query=query,
            response=response,
            domain=self._domain,
            timestamp=time.time(),
        )
        vec = self._embed(query).reshape(1, -1)
        self._index.add(vec)
        self._solutions.append(solution)
        self._maybe_prune()
        self._save_index()

    # --- embedding (lazy model load) ---

    def _embed(self, text: str) -> np.ndarray:
        if self._model is None:
            self._model = self._load_embedding_model()
        return self._model.encode(text, convert_to_numpy=True).astype(np.float32)

    @staticmethod
    def _load_embedding_model():
        """Lazy import + load of sentence-transformers model (CPU, 53 MB).

        Loads from pre-downloaded local copy to avoid HuggingFace network
        requests and noisy download progress bars at runtime.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

        # Pre-downloaded local model (created by setup or first download)
        local_model = (
            Path(__file__).resolve().parent / "embedding_model"
        )
        if local_model.exists():
            import io
            import contextlib
            import logging
            import os

            # Suppress HF hub noise, tokenizer warnings, progress bars
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

            # Silence the safetensors "Loading weights" tqdm bar and
            # the "BertModel LOAD REPORT" table (both go to stderr)
            with contextlib.redirect_stderr(io.StringIO()):
                return SentenceTransformer(
                    str(local_model), device="cpu", local_files_only=True,
                )

        # Fallback: download from HuggingFace and save locally for next time
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        local_model.mkdir(parents=True, exist_ok=True)
        model.save(str(local_model))
        return model

    # --- pruning ---

    def _maybe_prune(self) -> None:
        """If over max_vectors, rebuild index keeping most recent entries."""
        if len(self._solutions) <= self._max_vectors:
            return

        # sort by timestamp descending, keep most recent
        sorted_solutions = sorted(
            self._solutions, key=lambda s: s.timestamp, reverse=True,
        )
        self._solutions = sorted_solutions[:self._max_vectors]

        # rebuild FAISS index from scratch
        import faiss
        self._index = faiss.IndexFlatL2(self._faiss_dims)
        vectors = np.array(
            [self._embed(s.query) for s in self._solutions], dtype=np.float32,
        )
        self._index.add(vectors)

    # --- persistence (external I/O — try-except acceptable) ---

    def _index_path(self) -> Path:
        return self._memory_dir / f"{self._domain}.faiss"

    def _meta_path(self) -> Path:
        return self._memory_dir / f"{self._domain}.json"

    def _load_index(self) -> None:
        """Load existing FAISS index + metadata from disk, or create empty."""
        import faiss

        idx_path = self._index_path()
        meta_path = self._meta_path()

        if idx_path.exists() and meta_path.exists():
            try:
                self._index = faiss.read_index(str(idx_path))
                raw = json.loads(meta_path.read_text())
                self._solutions = [
                    Solution(
                        query=s["query"],
                        response=s["response"],
                        domain=s["domain"],
                        timestamp=s["timestamp"],
                    )
                    for s in raw
                ]
                return
            except (json.JSONDecodeError, OSError, KeyError):
                pass  # fall through to create empty

        self._index = faiss.IndexFlatL2(self._faiss_dims)
        self._solutions = []

    def _save_index(self) -> None:
        """Persist FAISS index + metadata to disk with atomic JSON write."""
        import faiss

        try:
            faiss.write_index(self._index, str(self._index_path()))
        except OSError:
            return

        meta = [
            {
                "query": s.query,
                "response": s.response,
                "domain": s.domain,
                "timestamp": s.timestamp,
            }
            for s in self._solutions
        ]
        tmp_path = self._meta_path().with_suffix(".tmp")
        try:
            tmp_path.write_text(json.dumps(meta, indent=2))
            os.replace(tmp_path, self._meta_path())
        except OSError:
            pass


# --- built-in test ---

if __name__ == "__main__" and "--test-persistence" in sys.argv:
    import tempfile

    print("=== AgentMemory Persistence Test ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Phase 1: store a solution
        mem1 = AgentMemory("test", tmpdir)
        mem1.store("How to find large files?", "find / -size +100M")
        print(f"  Stored 1 solution (index has {mem1.count} vectors)")

        # Phase 2: simulate restart — new instance, same directory
        mem2 = AgentMemory("test", tmpdir)
        print(f"  Reloaded index ({mem2.count} vectors)")

        # Phase 3: semantic search with different wording
        hits = mem2.search("find big files on linux")
        if hits:
            print(f"  Search returned {len(hits)} hit(s)")
            print(f"  Best hit: {hits[0].response!r}")

            # Verify the stored solution was retrieved
            match = any("find / -size +100M" in h.response for h in hits)
            status = "PASS" if match else "FAIL"
            print(f"  Match: {status}")
        else:
            print("  FAIL: no results returned")

        # Phase 4: verify L2 distance (cosine sim > 0.8)
        vec_q = mem2._embed("How to find large files?")
        vec_s = mem2._embed("find big files on linux")
        l2_sq = float(np.sum((vec_q - vec_s) ** 2))
        cosine_sim = 1.0 - l2_sq / 2.0
        status = "PASS" if cosine_sim > 0.8 else "FAIL"
        print(f"  Cosine similarity: {cosine_sim:.4f} ({status})")

    print("=== Done ===")
