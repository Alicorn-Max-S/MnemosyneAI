# Mnemosyne — Spec (Phases 1–3)

AI agent memory system. Phase 1: foundation (SQLite, FTS5, embeddings, Zvec). Phase 2: async write pipeline (Deriver via DeepSeek V3.2). Phase 3: retrieval pipeline (parallel search, RRF fusion, scoring, MMR dedup).

---

## Project Structure

```
mnemosyne/
├── pyproject.toml
├── CLAUDE.md
├── SPEC.md                     ← this file
├── mnemosyne/
│   ├── __init__.py
│   ├── config.py               # All constants, thresholds, API config
│   ├── models.py               # Pydantic: Peer, Session, Message, Note, Link, TaskItem, RetrievalResult
│   ├── db/
│   │   └── sqlite_store.py     # All SQLite CRUD, FTS5 search, task queue, access tracking
│   ├── vectors/
│   │   ├── zvec_store.py       # Zvec insert/search/delete wrapper
│   │   └── embedder.py         # nomic-embed-text-v1.5 (ONNX preferred, PyTorch fallback)
│   ├── api/
│   │   └── memory_api.py       # Public API coordinating all stores
│   ├── pipeline/               # Phase 2: write path
│   │   ├── __init__.py         # create_worker() factory
│   │   ├── intake.py           # Fast-path message ingestion (<5ms)
│   │   ├── deriver.py          # Extractor + Scorer (DeepSeek V3.2 via NousResearch)
│   │   ├── handlers.py         # handle_derive: extract → score → embed → store
│   │   └── worker.py           # Queue polling loop + dispatch
│   ├── retrieval/              # Phase 3: read path
│   │   ├── __init__.py         # create_retriever() factory
│   │   ├── scorer.py           # Pure functions: decay, provenance weight, fatigue, composite
│   │   ├── fusion.py           # RRF fusion + MMR dedup
│   │   └── retriever.py        # Orchestrator: parallel search → fuse → score → dedup → return
│   └── utils/
│       └── ids.py              # ULID generation
└── tests/
    ├── conftest.py
    ├── test_sqlite_store.py
    ├── test_zvec_store.py
    ├── test_embedder.py
    ├── test_memory_api.py
    ├── test_worker.py
    ├── test_intake.py
    ├── test_deriver.py
    ├── test_write_pipeline.py      # Phase 2 integration
    ├── test_scorer.py              # Phase 3: pure unit tests
    ├── test_fusion.py              # Phase 3: RRF + MMR
    ├── test_retriever.py           # Phase 3: integration with real stores
    └── test_retrieval_pipeline.py  # Phase 3: end-to-end write-then-read
```

---

## Dependencies

```toml
[project]
name = "mnemosyne"
version = "0.3.0"
requires-python = ">=3.11"
dependencies = [
    "aiosqlite>=0.20.0",
    "zvec>=0.2.0",
    "sentence-transformers>=5.0.0",
    "onnxruntime>=1.18.0",
    "pydantic>=2.0.0",
    "python-ulid>=3.0.0",
    "httpx>=0.27.0",
    "numpy>=1.26.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-asyncio>=0.23.0"]
```

Do NOT add fastapi, uvicorn, ragatouille, networkx, or google-genai yet. Those are Phase 4+.

numpy is added in Phase 3 for MMR cosine similarity (already a transitive dep of sentence-transformers, but pinned explicitly since fusion.py imports it directly).

---

## config.py

All constants live here. Key additions per phase are marked.

```python
# Storage
DEFAULT_DATA_DIR = "./mnemosyne_data"
SQLITE_DB_NAME = "mnemosyne.db"
ZVEC_COLLECTION_DIR = "zvec_notes"
ZVEC_COLLECTION_NAME = "mnemosyne_notes"

# Embedding
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 384
EMBEDDING_QUERY_PREFIX = "search_query: "
EMBEDDING_DOC_PREFIX = "search_document: "

# Schema
SCHEMA_VERSION = 1
RRF_K = 60

# Provenance values
PROVENANCE_ORGANIC = "organic"
PROVENANCE_AGENT_PROMPTED = "agent_prompted"
PROVENANCE_USER_CONFIRMED = "user_confirmed"
PROVENANCE_INFERRED = "inferred"

# Note types / durability
NOTE_TYPE_OBSERVATION = "observation"
NOTE_TYPE_INFERENCE = "inference"
DURABILITY_PERMANENT = "permanent"
DURABILITY_CONTEXTUAL = "contextual"
DURABILITY_EPHEMERAL = "ephemeral"

# Link types
LINK_TYPES = ["semantic", "causal", "temporal", "contradicts", "supports", "derived_from"]

# Defaults
DEFAULT_CONFIDENCE_OBSERVATION = 0.8
DEFAULT_CONFIDENCE_INFERENCE = 0.6
DEFAULT_CONFIDENCE_USER_SET = 1.0
DEFAULT_EMOTIONAL_WEIGHT = 0.5

# --- Phase 2: Deriver API ---
NOUSRESEARCH_BASE_URL = "https://portal.nousresearch.com/api-docs"
NOUSRESEARCH_MODEL = "deepseek/deepseek-v3.2"
DERIVER_EXTRACT_TEMPERATURE = 0.1
DERIVER_SCORE_TEMPERATURE = 0.1
DERIVER_MAX_RETRIES = 3
DERIVER_RETRY_DELAYS = [1.0, 2.0, 4.0]
WORKER_POLL_INTERVAL = 2.0

# --- Phase 3: Retrieval scoring ---
PROVENANCE_WEIGHTS = {"organic": 1.0, "user_confirmed": 0.8, "agent_prompted": 0.5, "inferred": 0.3}
DECAY_BASE_LAMBDA = 0.1          # Base decay rate
DECAY_IMPORTANCE_FACTOR = 0.8    # How much importance slows decay
DECAY_ACCESS_BOOST = 0.15        # Per-access strength boost
DECAY_MIN_MEMORIES = 100         # Decay disabled below this count
DECAY_RAMP_MAX = 1000            # Decay reaches full effect here
DECAY_HIGH_IMPORTANCE_FLOOR = 0.3  # Min strength for importance >= 0.7
SURFACING_FATIGUE_FACTOR = 0.1
MMR_SIMILARITY_THRESHOLD = 0.90
INFERENCE_SCORE_DISCOUNT = 0.7   # Inferences weighted lower than observations
RETRIEVAL_FTS_LIMIT = 30
RETRIEVAL_VECTOR_LIMIT = 30
RETRIEVAL_FINAL_LIMIT = 10
```

---

## SQLite Schema

PRAGMAs applied on every connection:
```sql
PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA cache_size=-8000;
PRAGMA busy_timeout=5000; PRAGMA mmap_size=268435456; PRAGMA temp_store=MEMORY;
PRAGMA foreign_keys=ON;
```

All PKs are TEXT (ULIDs). Timestamps: `strftime('%Y-%m-%dT%H:%M:%fZ', 'now')`.

**Tables:** config, peers, sessions, messages, notes, links, task_queue. See `sqlite_store.py` for full DDL. The critical table is **notes**:

| Column group | Columns |
|---|---|
| Identity | id, peer_id, session_id, source_message_id |
| Content | content, context_description, keywords (JSON), tags (JSON) |
| Classification | note_type, provenance, durability |
| Scoring | emotional_weight, importance, confidence |
| Frequency | evidence_count, unique_sessions_mentioned |
| Retrieval state | q_value, access_count, last_accessed_at, times_surfaced, decay_score |
| Pipeline state | is_buffered, canonical_note_id, zvec_id |
| Timestamps | created_at, updated_at |

**FTS5** virtual table on (content, context_description, keywords, tags) with sync triggers for INSERT/UPDATE/DELETE. Without triggers, FTS5 returns nothing.

---

## SQLiteStore Methods

**Phases 1–2** (already implemented): CRUD for peers/sessions/messages/notes/links. Task queue with atomic dequeue (UPDATE...RETURNING). `fts_search()` with BM25 scoring.

**Phase 3 additions:**

- `fts_search_ranked(query, peer_id, limit=30)` → note dicts with 0-indexed `fts_rank` positions (not just BM25 scores). RRF needs rank positions, not raw scores.
- `get_notes_by_ids(note_ids)` → fetch full rows for a list of IDs. Used to hydrate Zvec results from SQLite.
- `record_access(note_ids)` → batch UPDATE incrementing access_count, times_surfaced, and setting last_accessed_at. Single transaction. Called after retrieval returns final results.
- `count_notes(peer_id)` → integer count. Used by decay scoring to check if decay is active.

---

## Embedder / ZvecStore / MemoryAPI

Already implemented in Phases 1–2. See source files for details.

**Key facts for Phase 3:**
- `Embedder.embed_query(text)` prepends `"search_query: "`, returns 384-dim L2-normalized list[float].
- `Embedder.embed_documents(texts)` batch version for MMR embedding.
- `ZvecStore.search(query_embedding, top_k)` returns `[{"id": str, "score": float}, ...]`.
- Zvec has **no scalar fields/filtering** — returns results across all peers. Post-filter by peer_id after hydrating from SQLite.
- MemoryAPI gains `retrieve(query, peer_id, limit)` which delegates to the Retriever.

---

## Write Pipeline (Phase 2)

Already implemented. Brief summary for context:

1. **intake.py**: `ingest_message()` writes to SQLite + enqueues "derive" task for user messages. Assistant messages stored but not derived. < 5ms.
2. **deriver.py**: `Deriver.extract()` → atomic facts from user message. `Deriver.score()` → tags each note with emotional_weight, provenance, durability, keywords, tags, context_description. Both call DeepSeek V3.2 via NousResearch (httpx, not openai SDK). Retry 3x with 1/2/4s backoff.
3. **handlers.py**: `handle_derive()` wires extract → score → embed → store (SQLite + Zvec).
4. **worker.py**: polls task queue, dispatches to handlers, manages retries/dead-letter.

**Critical rules**: Agent responses are read for context but NEVER extracted from. User confirmations get `provenance: "user_confirmed"`. Notes stored with `is_buffered=1` for the Dreamer (Phase 5).

---

## Retrieval Pipeline (Phase 3)

No new external dependencies. No LLM calls. Pure local computation. Target: < 80ms end-to-end.

```
Query
  ├→ embed_query() ──→ Zvec search (top 30)
  ├→ FTS5 search (top 30)
  │         ↓
  │    Hydrate from SQLite + peer_id filter
  │         ↓
  │    RRF Fusion (k=60)
  │         ↓
  │    Score multipliers: decay × provenance × fatigue × inference_discount
  │         ↓
  │    MMR Dedup (cosine > 0.90 → drop lower-scored)
  │         ↓
  │    Top-N (default 10) → record_access() → return
```

### scorer.py — Pure scoring functions

All functions are stateless, no I/O. Take note metadata, return floats.

**`compute_decay_strength(importance, days_since_access, access_count, total_memories) → float`**
- Formula: `strength = importance * exp(-lambda_eff * days) * (1 + 0.15 * access_count)` where `lambda_eff = 0.1 * (1 - importance * 0.8)`
- Disabled (returns 1.0) when total_memories < 100
- Ramps linearly between 100–1000 memories: `ramp = min(1.0, total / 1000)`
- Floor of 0.3 for importance >= 0.7
- Clamped to [0.0, 1.0]

**`compute_provenance_weight(provenance) → float`**
- Lookup from `PROVENANCE_WEIGHTS`. Unknown → 0.5.

**`compute_surfacing_fatigue(times_surfaced) → float`**
- Formula: `1 / (1 + 0.1 * times_surfaced)`

**`compute_inference_discount(note_type) → float`**
- "inference" → 0.7, else → 1.0

**`compute_composite_score(rrf_score, decay, provenance, fatigue, inference_discount) → float`**
- Multiply all factors together.

### fusion.py — RRF + MMR

**`rrf_fuse(ranked_lists: list[list[str]], k=60) → dict[str, float]`**
- For each note across all lists: `score += 1 / (k + rank)` (0-indexed).
- Notes in multiple lists get multiple contributions.

**`mmr_dedup(scored_ids, embeddings: dict[str, list[float]], threshold=0.90) → list[str]`**
- Walk sorted IDs. Skip any candidate with cosine > threshold to an already-accepted result.
- nomic embeddings are L2-normalized, so cosine = dot product (use `np.dot`).
- Missing embeddings → auto-accept the note.

### retriever.py — Orchestrator

**`Retriever.__init__(db, zvec, embedder)`** — stores references.

**`Retriever.retrieve(query, peer_id, limit=10) → list[RetrievalResult]`**

Steps:
1. **Parallel search**: FTS5 (async) overlapped with embed_query (sync ~10ms) + Zvec search (sync ~2-5ms).
2. **Hydrate**: collect all unique IDs → `db.get_notes_by_ids()` → filter by peer_id.
3. **RRF fusion**: build two ranked ID lists → `rrf_fuse()`. Track source ("fts"/"vector"/"both").
4. **Score**: `db.count_notes(peer_id)`. For each note compute days_since_access (from last_accessed_at or created_at), then call all scorer functions → composite score.
5. **Sort + MMR dedup**: sort by composite → batch-embed candidates via `embedder.embed_documents()` → `mmr_dedup()`. Cap on-the-fly embedding to ≤ 10 FTS-only notes to avoid latency spikes.
6. **Truncate** to `limit`.
7. **Record access**: `db.record_access(final_ids)` — best-effort, never blocks.
8. **Return** `list[RetrievalResult]` sorted by score descending.

**Error handling**: Zvec fails → FTS-only. FTS fails → vector-only. Both fail → empty list. record_access failure → log, still return.

### RetrievalResult (models.py)

```python
class RetrievalResult(BaseModel):
    note: Note
    score: float              # Final composite score
    rrf_score: float          # Raw RRF before multipliers
    decay_strength: float
    provenance_weight: float
    fatigue_factor: float
    source: str               # "fts" | "vector" | "both"
```

### retrieval/__init__.py

```python
def create_retriever(db, zvec, embedder) -> Retriever:
    return Retriever(db=db, zvec=zvec, embedder=embedder)
```

---

## Latency Targets

| Path | Target | Notes |
|------|--------|-------|
| Fast path (write + enqueue) | < 5ms | User never waits past this |
| Full async write pipeline | < 5s | Not user-facing |
| FTS5 search | < 1ms | SQLite in-process |
| Zvec vector search | ~2-5ms | HNSW in-process |
| Query embedding | ~10ms | Single call |
| RRF + scoring | < 1ms | Pure math, ~60 candidates |
| MMR dedup | ~20-50ms | Batch embed + dot products |
| **Full retrieval** | **< 80ms** | **User-facing** |

---

## Environment Variables

```bash
export NOUSRESEARCH_API_KEY="your-key"   # Required for Phase 2 Deriver
```

No new env vars for Phase 3.

---

## Tests

All use `tmp_path`. All API calls mocked.

### Phase 1
- **test_sqlite_store.py**: schema, CRUD, FTS5 triggers, task queue atomic dequeue, dead-letter.
- **test_embedder.py**: 384-dim, deterministic, doc vs query prefix produces different vectors.
- **test_zvec_store.py**: insert+query, batch, delete, empty query.
- **test_memory_api.py**: full round trip peer→session→message→note→search (keyword, vector, hybrid).

### Phase 2
- **test_worker.py**: run_once, empty queue, exception handling, dead-letter.
- **test_intake.py**: user enqueues, assistant doesn't, preceding context, FTS5 findable.
- **test_deriver.py**: extraction, confirmation, empty, scorer fields, retry on 429, garbage JSON.
- **test_write_pipeline.py**: full round trip ingest→derive→notes in SQLite+Zvec+FTS5.

### Phase 3
- **test_scorer.py**: Pure unit tests. decay disabled < 100 memories. decay ramps 100–1000. high-importance floor. provenance weights match config. fatigue monotonically decreasing. composite multiplies all factors.
- **test_fusion.py**: RRF scores overlapping > non-overlapping. MMR drops similarity > 0.90, keeps orthogonal. Order preserved. Missing embeddings auto-accepted.
- **test_retriever.py**: basic retrieval, FTS+vector overlap gets source="both", peer isolation, access tracking updates, empty results, Zvec failure fallback, FTS failure fallback, MMR dedup of near-identical notes, provenance ordering, decay ordering, result limit respected.
- **test_retrieval_pipeline.py**: end-to-end write-then-read. Multiple sessions. Surfacing fatigue increases across repeated queries. Mixed provenance ordering.

---

## Design Principles

1. **User never waits for memory processing.** Write path is async; read path is < 80ms.
2. **Raw data is immutable.** Stream 1 observations are ground truth.
3. **Frequency ≠ importance.** Two-dimensional scoring prevents topic domination.
4. **Decay scores, never deletes.** All data preserved; relevance is temporal.
5. **Agent responses are context, not memory.** Read but never stored as user attributes.
6. **Graceful degradation.** If one search backend fails, the other still returns results.
7. **No LLM calls in the read path** (Phase 3). ColBERT/MemR3 are Phase 4+.
