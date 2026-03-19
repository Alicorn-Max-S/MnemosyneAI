# Mnemosyne: Complete Architecture Specification

## What this document is

This is the complete architectural specification for **Mnemosyne**, a next-generation AI agent memory system. It was designed over an extensive brainstorming session covering research into Honcho, A-MEM, MemR3, ColBERT, MAGMA, MemRL, Ebbinghaus decay, and commercial memory systems (ChatGPT, Claude, Gemini). Every decision below has been researched, debated, and agreed upon. The goal of the next conversation is to plan and build the implementation using Claude Code.

## Target deployment

- **Hostinger VPS**: 4 vCPU, 16GB RAM, Ubuntu 24.04, Docker installed
- **No local GPU** — all ML inference runs on CPU
- **Fine-tuning happens elsewhere** (RTX 3090 desktop), but the deployed system is CPU-only
- **Total RAM budget: ~2.6 GB** (84% of 16GB free)
- This is a general-purpose AI agent memory system, not student-specific

---

## Core storage layer

### Primary stores (all embedded, no external servers)

1. **SQLite** — source of truth for all structured data: notes, sessions, peers, links, config, task queue, static profile, graph adjacency tables. Uses WAL mode for concurrency. FTS5 virtual table for keyword search.

2. **Zvec** — embedded vector database (by Alibaba, Apache 2.0) for fast similarity search. Stores 384-dim embeddings with HNSW index. In-process, no server daemon. Supports scalar filtering, multi-vector retrieval, and built-in RRF reranking.

3. **No Postgres.** The entire point is replacing Postgres+pgvector with SQLite+Zvec to save 3-4GB RAM and eliminate server overhead.

### Key SQLite PRAGMAs
```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -8000;
PRAGMA busy_timeout = 5000;
PRAGMA mmap_size = 268435456;
PRAGMA temp_store = MEMORY;
```

---

## Three data streams

This is a critical architectural decision. Data exists in three forms:

### Stream 1: Raw observations (ground truth)
- Atomic fact notes extracted by the Deriver
- Stored in both SQLite (text + metadata) and Zvec (embeddings)
- **Never modified by the Dreamer** — this is immutable ground truth
- Primary search target during retrieval

### Stream 2: Linked graph (organized facts)
- A-MEM Zettelkasten-style bidirectional links between raw notes
- Stored as an adjacency table in SQLite
- Links have types: semantic, causal, temporal, contradicts, supports, derived_from
- Links have strength scores
- Used for graph expansion during retrieval ("box walk")

### Stream 3: Evolved knowledge (Dreamer output)
- Conclusions, patterns, inferences, Peer Card updates
- Stored as separate entries in Zvec tagged `type: INFERENCE`
- Lower default confidence than raw observations
- Searchable but weighted lower during retrieval
- The Dreamer can be aggressive here without risking raw data corruption

---

## Write path

### Fast path (synchronous, ~5ms, user never waits)
1. Message arrives
2. Written to SQLite + FTS5 immediately
3. Queued for async processing
4. User gets response instantly

### Deriver (async, two sequential API calls)

**Call 1 — Extractor (DeepSeek R1, off-peak)**
- Input: the user's full message + 2-3 preceding turns (both user and agent) for context
- Job: extract atomic fact notes from the user's message only
- Agent responses are READ for context but NEVER extracted from
- Exception: if user confirms something agent said ("yeah that's right"), attribute the confirmed fact to the user with `provenance: user_confirmed`
- Output: list of atomic observation notes
- Cost estimate: ~1,000 input + ~1,000 thinking + ~200 output = ~$0.00012 off-peak

**Call 2 — Scorer (DeepSeek V3 Chat, cheaper)**
- Input: just the extracted notes (short text)
- Job: tag each note with:
  - `emotional_weight` (0.0-1.0): detected from language signals — possessives, named entities, strong verbs/adjectives, temporal attachment ("6 years") = high weight
  - `provenance`: organic | agent_prompted | user_confirmed | inferred
  - `durability`: permanent (facts about the person), contextual (current projects), ephemeral (one-off tasks)
  - `keywords`: for A-MEM note construction
  - `tags`: categorical labels
  - `context_description`: brief summary of conversational context
- Cost estimate: ~300 input + ~500 thinking + ~150 output = ~$0.00005 off-peak

**After scoring:**
- Notes are embedded using nomic-embed-text-v1.5 on CPU (~10ms per note)
- Embeddings stored in Zvec (Stream 1)
- Note text + metadata stored in SQLite
- Notes added to the raw buffer for later batch dedup

### Raw note buffer
- Notes accumulate in SQLite for hours/days between Dreamer cycles
- No processing happens here — just accumulation

### Batch dedup (runs once per Dreamer cycle, ~100ms)
- When the Dreamer wakes up, FIRST run batch dedup on the buffer
- Cluster notes by embedding similarity > 0.85 (pure math, no LLM)
- For each cluster: merge into one canonical note, set `evidence_count` = cluster size
- Track `unique_sessions_mentioned` (user-originated mentions only, one count per session max)
- This is the gateway to the Dreamer — it only sees clean, deduplicated notes

### Importance formula (two-dimensional)
```
importance = emotional_weight × 0.6 + frequency_score × 0.4
frequency_score = 1 - e^(-0.15 × unique_sessions_mentioned)
```

The exponential curve means sessions 1-5 matter most, 5-15 matter somewhat, 15+ barely moves the needle. This ensures a topic mentioned in 50 sessions doesn't dominate over something mentioned once with high emotional weight.

### Dreamer (background process, DeepSeek batch API)
- Receives clean, weighted, deduplicated notes
- Performs:
  - **Link generation**: find neighbors for new notes, create typed bidirectional links (Stream 2)
  - **Pattern detection**: cross-session trends, behavioral patterns
  - **Contradiction detection**: flag conflicting observations (NLI check with cosine > 0.80 pairs)
  - **Profile update**: update the static Peer Card when durable facts change
  - **MAGMA multi-graph update**: entity graph, temporal graph, causal graph, contradiction log
- Outputs Stream 3 entries (tagged `type: INFERENCE`)
- Uses NetworkX in-memory for graph algorithms, persisted to SQLite

### Durability classification
- `permanent` (importance preserved, minimal decay): "has a cat named Buddy"
- `contextual` (moderate decay, ~7-14 day half-life): "working on a memory system project"
- `ephemeral` (importance=0.0, decays to nothing in 2-3 days): "remind me to buy milk"

---

## Read path

### Always present: Static profile (~30 facts, ~400 tokens)
- Injected at the TOP of every system prompt (benefits from primacy bias + prompt caching)
- Four sections only: identity, professional (role/company, NOT skill level), communication style, relationships
- **NO expertise/skill level in static profile** — this causes models to over-anchor and treat it as gospel. Expertise lives in dynamic memory where it can evolve and only surfaces when relevant.
- User-set facts override LLM observations (confidence 1.0 vs 0.8)
- Generated by the Dreamer, editable by user

### Parallel search (~5ms total)
Three searches run simultaneously:

1. **FTS5 keyword search** (SQLite): BM25 ranking, <1ms
2. **Zvec vector search**: cosine similarity over Stream 1 (raw) + Stream 3 (inferences, weighted lower), ~2-5ms
3. **A-MEM link expansion**: for top vector results, walk links in Stream 2 to find connected notes, ~1ms

### MMR dedup on results
- After combining results from all three searches (~50 candidates)
- If two results have cosine similarity > 0.90 to each other, keep the higher-scored one
- Ensures retrieval slots carry diverse information, not paraphrases

### RRF fusion + scoring
Combine results using Reciprocal Rank Fusion with these scoring multipliers:
- **Ebbinghaus decay** (scoring only, never deletes data):
  ```
  lambda_eff = 0.1 * (1 - importance * 0.8)
  strength = importance * exp(-lambda_eff * days_since_access) * (1 + 0.15 * access_count)
  ```
- **MemRL Q-value**: learned utility score per memory, updated by user feedback
- **Provenance weight**: organic=1.0, user_confirmed=0.8, agent_prompted=0.5, inferred=0.3
- **Surfacing fatigue**: `1 / (1 + 0.1 × times_surfaced_in_recent_N_sessions)`

### ColBERT reranking (~20ms on CPU)
- Model: answerai-colbert-small-v1 (33M params, 96-dim, ~130MB)
- Pre-computed token embeddings stored per memory (~4KB each with binarization)
- Reranks top 20-50 candidates from fusion step → outputs top 10
- With pre-computed embeddings, latency is nearly independent of candidate count
- Only query encoding requires a forward pass; MaxSim scoring is pure matrix ops

### Router: simple vs complex
- **Simple (88% of queries, ~50ms total)**: direct answer from top-10 results
- **Complex (12%, ~2-4s)**: triggers MemR3 reflective loop
- Routing signals: embedding distance distribution, query structure heuristics, score spread

### MemR3 reflective loop (complex queries only)
- Maintains evidence-gap tracker: what's known vs what's missing
- Each iteration: retrieve → reflect → decide (retrieve again, reflect more, or answer)
- Mask previously retrieved snippets to prevent re-retrieval
- Most complex queries resolve in 1-3 iterations
- Can use a cheaper model for routing/reflection, stronger model for final answer

### MemRL Q-value updates (after every response)
- Track which memories were retrieved and injected
- User feedback flows into Q-values:
  - Thumbs up: `Q_new = (1-0.1) × Q_old + 0.1 × 1.0`
  - Thumbs down: `Q_new = (1-0.1) × Q_old + 0.1 × -0.5` (asymmetric)
  - Conversational correction: invalidate contradicted memory, create replacement
  - Implicit: topic change = weak negative (±0.05), long engaged response = weak positive
- Q-values stored as single float per memory in SQLite

---

## Anti-sycophancy defenses (5 layers)

This is critical. MIT's February 2026 study proved that condensed user profiles increase LLM sycophancy. Memory-powered personalization creates self-reinforcing feedback loops if undefended.

1. **Provenance tagging**: every memory tagged organic vs agent_prompted, weighted differently at retrieval
2. **Provenance-weighted scoring**: organic=1.0, agent_prompted=0.5 in retrieval scoring
3. **MMR diversity enforcement**: no more than 40% of injected memories from same topic cluster
4. **Epsilon-greedy exploration**: with probability 0.1-0.15, replace one retrieved memory with a random selection
5. **Anti-sycophancy system prompt**: appended when memory context is injected, instructing the model not to over-index on memory or introduce topics merely because they appear in memory

### Agent response handling
- Agent responses are **read by the Extractor for context** but **never extracted from**
- Exception: user confirmation ("yeah that's right") → attribute to user as `provenance: user_confirmed`
- If the same agent-introduced topic appears in 3+ consecutive sessions without organic user mention, flag associated memories for decay

---

## Ebbinghaus decay details

- **Scoring only — never deletes data**
- Raw data stays in SQLite forever (or moves to cold tier)
- Decay just means stale memories rank lower in retrieval
- **Disabled below 100 memories** (too little data for decay to be useful)
- Ramps linearly: `effective_decay = base_decay × min(1.0, memory_count / 1000)`
- High-importance memories have a floor score of 0.3 regardless of decay
- Each retrieval resets the access clock and increments recall count (spaced repetition)

### Tiered storage via SQLite ATTACH
- **Hot** (main.db): decay_score > 0.2 OR accessed within 30 days. Full FTS5 + vector index.
- **Warm** (warm.db): 0.05 < decay_score ≤ 0.2. FTS5 only.
- **Cold** (archive.db): everything else. Minimal indexing, searched only on deep-retrieval requests.
- Daily maintenance job migrates between tiers

---

## RAM budget (conservative estimates at 100K memories)

| Component | RAM |
|-----------|-----|
| Ubuntu 24.04 + Docker | 700 MB |
| Python + PyTorch + libs | 500 MB |
| nomic-embed-text-v1.5 (F32) | 600 MB |
| answerai-colbert-small-v1 | 200 MB |
| Zvec HNSW (100K × 384-dim) | 200 MB |
| SQLite + FTS5 (100K records) | 50 MB |
| NetworkX graph (100K nodes) | 150 MB |
| Working memory / buffers | 200 MB |
| **Total** | **~2,600 MB** |
| **Free from 16 GB** | **~13,400 MB (84%)** |

---

## API pricing (async Deriver)

| Provider | Model | Use case | Input $/M | Output $/M |
|----------|-------|----------|-----------|------------|
| DeepSeek (off-peak) | R1 Reasoner | Extractor (Call 1) | $0.07 | $0.105 |
| DeepSeek (off-peak) | V3.2 Chat | Scorer (Call 2) | $0.07 | $0.105 |
| DeepSeek (standard) | V3.2 Reasoner | Fallback | $0.28 | $0.42 |
| Cerebras | Various | Fast inference option | Variable | Variable |
| Google Batch | Gemini 2.5 Flash | Dreamer batch | $0.15 | $1.25 |

Estimated cost per user message: ~$0.00017 (Deriver) + negligible retrieval = **well under $0.001/message**

---

## ML models running on CPU

1. **nomic-embed-text-v1.5** (137M params, 768-dim with Matryoshka support, truncate to 384): embedding model for all notes and queries. ~600MB RAM, ~10ms per embedding on CPU. Apache 2.0, local inference. Can be fine-tuned later on synthetic data with MatryoshkaLoss.

2. **answerai-colbert-small-v1** (33M params, 96-dim tokens): ColBERT reranker. ~130MB model, ~200MB loaded. Pre-compute token embeddings at write time, MaxSim scoring at read time. ~20ms reranking on CPU.

Both should use ONNX Runtime with INT8 quantization for ~3× CPU speedup.

---

## Build phases

### Phase 1 — Foundation (get data flowing)
- SQLite schema (notes, sessions, peers, links, config, queue, profile)
- FTS5 virtual table
- Basic CRUD API in Python (FastAPI or functions)
- nomic-embed-text running on CPU
- Zvec collection creation and basic insert/query

### Phase 2 — Write pipeline
- Deriver Call 1 (Extractor) via DeepSeek API
- Deriver Call 2 (Scorer) via DeepSeek V3
- Async task queue (SQLite-based)
- Raw notes → SQLite + FTS5 + Zvec simultaneously
- Provenance tagging, durability classification, emotional weight

### Phase 3 — Basic retrieval
- FTS5 keyword search
- Zvec vector search
- RRF fusion
- MMR dedup on results
- Ebbinghaus decay scoring in SQL query

### Phase 4 — Intelligence layer
- A-MEM link generation + graph storage
- Link expansion during retrieval (third search stream)
- ColBERT reranking on candidates
- Static profile generation and injection

### Phase 5 — Background processes
- Batch dedup (clustering + merge + importance scoring)
- Dreamer (pattern detection, contradiction check, profile updates)
- MAGMA multi-graph (entity, temporal, causal)

### Phase 6 — Refinement
- MemR3 reflective loop for complex queries
- MemRL Q-value tracking + user feedback
- Anti-sycophancy guards (provenance weighting, epsilon exploration, fatigue)
- Adaptive retrieval depth routing
- User feedback ingestion and memory correction flow

---

## Key research references

- **Honcho** (Plastic Labs): three-stage reasoning pipeline, Peer Cards, dialectic retrieval. AGPL-3.0. github.com/plastic-labs/honcho
- **A-MEM** (NeurIPS 2025): Zettelkasten-style linked notes, memory evolution. MIT. github.com/agiresearch/A-mem
- **MemR3** (arXiv:2512.20237): reflective retrieval with evidence-gap tracking. +7.29% on LoCoMo.
- **ColBERT** / answerai-colbert-small-v1: late interaction reranking. 33M params, outperforms 110M ColBERTv2. github.com/AnswerDotAI/RAGatouille
- **MAGMA** (arXiv:2601.03236): multi-graph memory with fast/slow path separation.
- **MemRL** (arXiv:2601.03192): Q-value based memory selection via RL. github.com/anvanster/tempera
- **Zvec** (Alibaba): embedded vector DB. >8,000 QPS on 10M dataset. Apache 2.0. github.com/alibaba/zvec
- **YourMemory**: Ebbinghaus decay achieving 34% vs Mem0's 18% Recall@5.
- **MIT sycophancy study** (February 2026): proves condensed profiles increase LLM sycophancy.

---

## Recommended research for the implementing model

Before starting implementation, familiarize yourself with these resources in priority order. The architecture draws heavily from specific techniques in each — understanding them will prevent misimplementation.

### Must-read (core architecture depends on these)

1. **Honcho source code and docs** — The three-agent pipeline (Deriver/Dreamer/Dialectic) is the backbone we're adapting. Read the CLAUDE.md file for architecture overview, understand how observations are extracted, how dreaming/consolidation works, and how the dialectic agent performs agentic retrieval.
   - github.com/plastic-labs/honcho (especially CLAUDE.md and src/deriver/)
   - docs.honcho.dev/v3/documentation/introduction/overview
   - blog.plasticlabs.ai/blog/Honcho-3 (v3 architecture changes)
   - blog.plasticlabs.ai/research/Benchmarking-Honcho (benchmark methodology)

2. **A-MEM paper and implementation** — Our note structure, link generation, and "box expansion" retrieval come from here. Understand how notes are constructed (7 fields), how links are generated (embedding similarity → LLM evaluation), and how memory evolution updates existing notes.
   - arxiv.org/abs/2502.12110 (paper)
   - github.com/agiresearch/A-mem (official implementation, uses ChromaDB)
   - github.com/tobs-code/a-mem-mcp-server (third-party MCP server with NetworkX graph — closest to our architecture)

3. **Zvec documentation** — Our vector store. Understand the CollectionSchema API, HNSW index configuration, scalar filtering, multi-vector retrieval, and built-in RRF reranking.
   - zvec.org/en/docs/ (quickstart and API reference)
   - github.com/alibaba/zvec (README has complete code examples)
   - zvec.org/en/blog/introduction/ (architecture whitepaper)

4. **SQLite FTS5 + hybrid search** — Our keyword search layer. Understand FTS5 virtual tables, BM25 ranking, and how to combine FTS5 results with vector search via RRF.
   - sqlite.org/fts5.html (official docs)
   - alexgarcia.xyz/blog/2024/sqlite-vec-hybrid-search/ (hybrid search with RRF in SQL — directly applicable pattern)

### Should-read (quality and refinement layers)

5. **ColBERT / answerai-colbert-small-v1** — Our reranking layer. Understand late interaction (MaxSim), why pre-computed token embeddings make CPU reranking fast, and storage tradeoffs.
   - answer.ai/posts/2024-08-13-small-but-mighty-colbert.html (model introduction)
   - answer.ai/posts/colbert-pooling.html (token pooling for storage reduction)
   - github.com/AnswerDotAI/RAGatouille (simplest API for ColBERT indexing/search)
   - pypi.org/project/pylate/ (flexible training and retrieval, sentence-transformers compatible)

6. **MemR3 paper** — Our reflective retrieval loop for complex queries. Understand the evidence-gap tracker, the retrieve/reflect/answer routing, and the mask mechanism.
   - arxiv.org/abs/2512.20237 (paper)
   - Key finding: most queries resolve in 1 iteration; masking previously-retrieved snippets is the most important component

7. **MemRL paper and Tempera implementation** — Our Q-value system for learning which memories are useful. Understand the two-phase retrieval (similarity filter → Q-value reranking), Bellman updates, and z-score normalization.
   - arxiv.org/abs/2601.03192 (paper, titled "Self-Evolving Agents via Runtime RL on Episodic Memory")
   - github.com/anvanster/tempera (Rust MCP server implementation — study the Q-value update logic)

8. **MAGMA paper** — Our multi-graph secondary storage design. Understand the four orthogonal graphs (semantic, temporal, causal, entity) and the fast-path/slow-path separation.
   - arxiv.org/abs/2601.03236 (paper)

### Good to know (context and alternatives)

9. **Ebbinghaus decay implementations** — Understand the forgetting curve formula and how it's been applied to AI memory.
   - dev.to/sachit_mishra — "I built memory decay for AI agents using the Ebbinghaus forgetting curve"
   - YourMemory MCP Server — importance-weighted decay formula we adapted
   - news.ycombinator.com/item?id=47070979 — Kore memory layer discussion

10. **Matryoshka embeddings** — If fine-tuning the embedding model later. Understand how MRL trains multi-resolution embeddings and the MatryoshkaLoss in sentence-transformers.
    - sbert.net/examples/sentence_transformer/training/matryoshka/README.html
    - arxiv.org/abs/2205.13147 (original MRL paper)

11. **Anti-sycophancy research** — Context for why our provenance tracking matters.
    - news.mit.edu/2026/personalization-features-can-make-llms-more-agreeable-0218 (the key study)
    - arxiv.org/abs/2503.03704 (MINJA memory injection attacks — why provenance tracking is also a security measure)

12. **Commercial memory implementations** — Context for what we're improving upon.
    - llmrefs.com/blog/reverse-engineering-chatgpt-memory (ChatGPT memory reverse-engineered)
    - simonwillison.net/2025/Sep/12/claude-memory/ (Claude vs ChatGPT memory comparison)
    - shloked.com/writing/gemini-memory (Gemini's deliberate restraint)

13. **Mem0 architecture** — The most popular open-source alternative. Understand what it does (ADD/UPDATE/DELETE/NOOP per fact, graph memory via Neo4j) and where it falls short (49% on LongMemEval, 3 LLM calls per write, 18% Recall@5).
    - deepwiki.com/mem0ai/mem0 (architecture overview)
    - arxiv.org/abs/2504.19413 (Mem0 paper with benchmark results)

14. **nomic-embed-text-v1.5** — Our embedding model. Understand its Matryoshka support (truncate 768→384→128→64), 8192 token context, and ONNX export for CPU speedup.
    - huggingface.co/nomic-ai/nomic-embed-text-v1.5
    - nomic.ai/news/nomic-embed-matryoshka

15. **SimpleMem** — An alternative approach worth understanding: front-loads work at ingestion via semantic compression, achieving strong results with only 531 tokens/query.
    - arxiv.org/abs/2601.02553

### Libraries and tools to install

```
# Core
pip install zvec sentence-transformers aiosqlite

# ColBERT reranking
pip install ragatouille  # or: pip install pylate

# Embedding model
pip install onnxruntime  # for 3x CPU speedup

# Graph
pip install networkx

# API clients
pip install httpx  # for DeepSeek API calls

# Web framework (if exposing as API)
pip install fastapi uvicorn
```

---

## Design principles

1. **User never waits for memory processing.** Everything after the SQLite write is async.
2. **Raw data is immutable.** The Dreamer only creates Stream 3 inferences, never modifies Stream 1.
3. **Frequency ≠ importance.** Two-dimensional scoring (emotional weight × frequency) prevents topic domination.
4. **Memory informs, doesn't dominate.** Anti-sycophancy guards prevent self-reinforcing loops.
5. **Decay scores, never deletes.** All data is preserved; relevance is temporal.
6. **Expertise stays dynamic.** Not in the static profile to prevent model over-anchoring.
7. **Agent responses are context, not memory.** Read but never stored as the user's attributes.
8. **Ephemeral things fade naturally.** Durability classification prevents task noise from polluting long-term memory.
