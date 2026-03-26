# Memory System Test Results (Step 5)

## Test 1: SharedState (Tier 1) — ALL PASS
- Persistence: PASS (set/get survives new instance)
- Action log: PASS (entries stored correctly)
- Snapshot: PASS (df/ip/ps return data)
- Cross-agent context: PASS
- Log cap: PASS (capped at 50 entries)

## Test 2: AgentMemory Persistence (Tier 2) — ALL PASS
- Store + restart + semantic search: PASS
- Best hit: 'find / -size +100M' (correct match)
- Cosine similarity: 0.8984 (threshold: > 0.8)
- L2^2 distance: well within 0.4 threshold

## Test 3: SessionContext (Tier 3) — ALL PASS
- Turn cap: PASS (5 max, oldest evicted)
- Oldest turn: PASS (correct eviction order)
- Context string: PASS (formats last N turns)
- Metadata: PASS (set/get works)
- Clear: PASS (resets everything)
- Empty context: PASS (returns "")

## Test 4: Routing Regression — NO REGRESSION
- Keywords only: 40/44 (90.9%) — identical to Step 4
- Full routing: 41/44 (93.2%) — identical to Step 4
- Same 3 failures as Step 4 (all in model fallback for ambiguous queries)

## Test 5: End-to-End with Memory — ALL PASS
- files: OK (find / -size +100M)
- network: OK (sudo ss -tlnp)
- process: OK (pkill -f nginx)
- packages: OK (sudo dpkg -i)
- kernel: OK (sudo modprobe)
- All 5 domains route and respond correctly with memory wired in
- memory_hits field present in all responses

## Known Observations
- Embedding model (all-MiniLM-L6-v2) loads separately per agent (~1s each, 53 MB)
  - Could share a single model instance across agents in future optimization
  - Not a blocker: model is small and loads quickly
- FAISS index created per domain in ~/.local/share/ai-daemon/memory/
- "embeddings.position_ids UNEXPECTED" warning from sentence-transformers is benign
