# Mnemosyne — Spec (Phases 1–4)

AI agent memory system. Phase 1: foundation (SQLite, FTS5, embeddings, Zvec). Phase 2: async write pipeline (Deriver via DeepSeek V3.2). Phase 3: retrieval pipeline (parallel search, RRF fusion, scoring, MMR dedup). Phase 4: intelligence layer (A-MEM link generation, link expansion, ColBERT reranking, static profile).

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
│   ├── models.py               # Pydantic: Peer, Session, Message, Note, Link, TaskItem,
│   │                           #   RetrievalResult, PeerProfile
│   ├── db/
│   │   └── sqlite_store.py     # All SQLite CRUD, FTS5 search, task queue, access tracking,
│   │                           #   link graph traversal, profile storage
│   ├── vectors/
│   │   ├── zvec_store.py       # Zvec insert/search/delete wrapper
│   │   └── embedder.py         # nomic-embed-text-v1.5 (ONNX preferred, PyTorch fallback)
│   ├── api/
│   │   └── memory_api.py       # Public API coordinating all stores
│   ├── pipeline/               # Phase 2: write path
│   │   ├── __init__.py         # create_worker() factory
│   │   ├── intake.py           # Fast-path message ingestion (<5ms)
│   │   ├── deriver.py          # Extractor + Scorer (DeepSeek V3.2 via NousResearch)
│   │   ├── handlers.py         # handle_derive: extract → score → embed → store → link
│   │   └── worker.py           # Queue polling loop + dispatch
│   ├── retrieval/              # Phase 3: read path
│   │   ├── __init__.py         # create_retriever() factory
│   │   ├── scorer.py           # Pure functions: decay, provenance weight, fatigue, composite
│   │   ├── fusion.py           # RRF fusion + MMR dedup
│   │   └── retriever.py        # Orchestrator: parallel search → link expand → fuse → score
│   │                           #   → ColBERT rerank → return
│   ├── intelligence/           # Phase 4: intelligence layer
│   │   ├── __init__.py         # create_linker(), create_reranker(), create_profiler() factories
│   │   ├── linker.py           # A-MEM link generation (embedding similarity, no LLM)
│   │   ├── reranker.py         # ColBERT reranker wrapper via rerankers library
│   │   └── profiler.py         # Static profile generation via Deriver API
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
    ├── test_models.py
    ├── test_scorer.py              # Phase 3: pure unit tests
    ├── test_fusion.py              # Phase 3: RRF + MMR
    ├── test_retriever.py           # Phase 3: integration with real stores
    ├── test_retrieval_pipeline.py  # Phase 3: end-to-end write-then-read
    ├── test_linker.py              # Phase 4: A-MEM link generation
    ├── test_reranker.py            # Phase 4: ColBERT reranker
    ├── test_profiler.py            # Phase 4: static profile generation
    ├── test_link_expansion.py      # Phase 4: link expansion in retrieval
    └── test_intelligence_pipeline.py  # Phase 4: end-to-end intelligence integration
```

---

## Dependencies

```toml
[project]
name = "mnemosyne"
version = "0.4.0"
requires-python = ">=3.11"
dependencies = [
    "aiosqlite>=0.20.0",
    "zvec>=0.2.0",
    "sentence-transformers[onnx]>=5.0.0",
    "onnxruntime>=1.18.0",
    "einops>=0.8.0",
    "pydantic>=2.0.0",
    "python-ulid>=3.0.0",
    "httpx>=0.27.0",
    "numpy>=1.26.0",
    "rerankers[transformers]>=0.10.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-asyncio>=0.23.0"]
```

Do NOT add fastapi, uvicorn, ragatouille, pylate, colbert-ai, networkx, or google-genai yet. Those are Phase 5+.

numpy is added in Phase 3 for MMR cosine similarity (already a transitive dep of sentence-transformers, but pinned explicitly since fusion.py imports it directly).

rerankers is added in Phase 4 for ColBERT reranking. The `[transformers]` extra is required for local model inference. Do NOT use `ragatouille` or `pylate` — the `rerankers` library provides the simplest API for reranking a small candidate set without managing a persistent ColBERT index.

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
NOUSRESEARCH_BASE_URL = "https://inference-api.nousresearch.com/v1"
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

# --- Phase 4: Intelligence layer ---

# ColBERT reranking
COLBERT_MODEL = "answerdotai/answerai-colbert-small-v1"
COLBERT_RERANK_CANDIDATES = 30      # Max candidates to rerank
COLBERT_TOP_N = 10                  # Return top N after reranking

# A-MEM link generation
LINK_SIMILARITY_THRESHOLD = 0.75    # Cosine similarity threshold for creating links
LINK_MAX_CANDIDATES = 10            # Max candidate notes to evaluate for linking
LINK_DEFAULT_STRENGTH = 0.5         # Default link strength
LINK_STRENGTH_FROM_SIMILARITY = True  # Map cosine similarity to link strength

# Link expansion in retrieval
LINK_EXPANSION_DEPTH = 1            # Walk links N hops (1 = direct neighbors only)
LINK_EXPANSION_MAX = 5              # Max notes to add from link expansion per seed note
LINK_EXPANSION_TOP_SEEDS = 5        # Number of top retrieval results to expand from

# Static profile
PROFILE_MAX_FACTS = 30              # Max facts in the static profile
PROFILE_MAX_TOKENS = 400            # Approximate token budget for the profile
PROFILE_SECTIONS = ["identity", "professional", "communication_style", "relationships"]
PROFILE_MIN_NOTES = 5               # Min permanent notes before generating a profile
PROFILE_REGENERATE_INTERVAL_HOURS = 24  # How often to regenerate the profile
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

**Tables:** config, peers, sessions, messages, notes, links, task_queue, peer_profiles. See `sqlite_store.py` for full DDL. The critical table is **notes**:

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

**peer_profiles table (Phase 4):**

```sql
CREATE TABLE IF NOT EXISTS peer_profiles (
    peer_id TEXT PRIMARY KEY REFERENCES peers(id),
    sections TEXT NOT NULL DEFAULT '{}',
    fact_count INTEGER NOT NULL DEFAULT 0,
    generated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    source_note_ids TEXT NOT NULL DEFAULT '[]'
);
```

**FTS5** virtual table on (content, context_description, keywords, tags) with sync triggers for INSERT/UPDATE/DELETE. Without triggers, FTS5 returns nothing.

---

## SQLiteStore Methods

### Phases 1–2 (already implemented)

CRUD for peers/sessions/messages/notes/links. Task queue with atomic dequeue (UPDATE...RETURNING). `fts_search()` with BM25 scoring.

### Phase 3 additions

- `fts_search_ranked(query, peer_id, limit=30)` → note dicts with 0-indexed `fts_rank` positions (not just BM25 scores). RRF needs rank positions, not raw scores.
- `get_notes_by_ids(note_ids)` → fetch full rows for a list of IDs. Used to hydrate Zvec results from SQLite.
- `record_access(note_ids)` → batch UPDATE incrementing access_count, times_surfaced, and setting last_accessed_at. Single transaction. Called after retrieval returns final results.
- `count_notes(peer_id)` → integer count. Used by decay scoring to check if decay is active.

### Phase 4 additions

- `get_linked_notes(note_ids: list[str], depth: int = 1, max_per_seed: int = 5) → list[Note]` — Walk the link graph from a set of seed note IDs. Returns notes that are linked (directly or up to `depth` hops) to any seed. Steps:
  1. Query the `links` table for rows where `source_note_id IN (?) OR target_note_id IN (?)`.
  2. Collect the "other side" note IDs that are NOT already in `note_ids`.
  3. Cap at `max_per_seed` per seed note.
  4. Fetch full `Note` rows via `get_notes_by_ids()`.
  5. If `depth > 1`, recurse with the new IDs (but never re-visit already-seen IDs).
  6. Return deduplicated list of Notes.

- `upsert_profile(peer_id: str, sections: dict, fact_count: int, source_note_ids: list[str]) → PeerProfile` — INSERT OR REPLACE into `peer_profiles`. Return the `PeerProfile` object.

- `get_profile(peer_id: str) → PeerProfile | None` — Fetch the profile for a peer. Return `None` if no profile exists.

- `get_permanent_notes(peer_id: str, limit: int = 50) → list[Note]` — Convenience method: `SELECT * FROM notes WHERE peer_id = ? AND durability = 'permanent' ORDER BY importance DESC, created_at DESC LIMIT ?`.

---

## Pydantic Models (models.py)

### Existing models (Phases 1–2)

Peer, Session, Message, Note, Link, TaskItem. All have `from_row()` class methods for aiosqlite Row dict construction. See source for full definitions.

### Phase 3 addition

```python
class RetrievalResult(BaseModel):
    note: Note
    score: float              # Final score (ColBERT if available, else composite)
    composite_score: float    # Pre-reranking composite score
    rrf_score: float          # Raw RRF before multipliers
    colbert_score: float | None = None  # ColBERT MaxSim score, None if ColBERT unavailable
    decay_strength: float
    provenance_weight: float
    fatigue_factor: float
    source: str               # "fts" | "vector" | "link" | "both" | "fts+link" | etc.
```

Note: the `colbert_score` and `composite_score` fields are Phase 4 additions to the Phase 3 model. Phase 3 implementation should include `composite_score` (same as `score` when no ColBERT) and `colbert_score: float | None = None`.

### Phase 4 addition

```python
class PeerProfile(BaseModel):
    """A static profile (Peer Card) generated from permanent notes."""

    peer_id: str
    sections: dict[str, str]  # section_name -> text content
    fact_count: int
    generated_at: str
    source_note_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_row(cls, row: dict) -> "PeerProfile":
        """Construct from an aiosqlite Row dict."""
        data = dict(row)
        if isinstance(data.get("sections"), str):
            data["sections"] = json.loads(data["sections"])
        if isinstance(data.get("source_note_ids"), str):
            data["source_note_ids"] = json.loads(data["source_note_ids"])
        return cls(**data)
```

---

## Embedder / ZvecStore / MemoryAPI

Already implemented in Phases 1–2. See source files for details.

**Key facts for Phases 3–4:**
- `Embedder.embed_query(text)` prepends `"search_query: "`, returns 384-dim L2-normalized list[float].
- `Embedder.embed_documents(texts)` batch version for MMR embedding.
- `Embedder.embed_document(text)` single document embedding with `"search_document: "` prefix.
- `ZvecStore.search(query_embedding, top_k)` returns `[{"id": str, "score": float}, ...]`.
- Zvec has **no scalar fields/filtering** — returns results across all peers. Post-filter by peer_id after hydrating from SQLite.
- For L2-normalized vectors (which nomic-embed produces), Zvec's score represents inner product = cosine similarity. Higher = more similar. Use the score directly for the link threshold comparison.
- MemoryAPI gains `retrieve(query, peer_id, limit)` which delegates to the Retriever.

---

## Write Pipeline (Phase 2, extended in Phase 4)

### Fast path (synchronous, ~5ms, user never waits)

1. **intake.py**: `ingest_message()` writes to SQLite + enqueues "derive" task for user messages. Assistant messages stored but not derived.

### Async pipeline

2. **deriver.py**: `Deriver.extract()` → atomic facts from user message. `Deriver.score()` → tags each note with emotional_weight, provenance, durability, keywords, tags, context_description. Both call DeepSeek V3.2 via NousResearch (httpx, not openai SDK). Retry 3x with 1/2/4s backoff.

3. **handlers.py**: `handle_derive()` wires extract → score → embed → store (SQLite + Zvec) → **link** (Phase 4).

4. **worker.py**: polls task queue, dispatches to handlers, manages retries/dead-letter.

### Phase 4 write path addition: Link generation

After creating each note and inserting into Zvec, the handler calls `linker.generate_links(note, embedding)`:

```python
# In handle_derive(), after db.create_note() and zvec.insert():
if linker is not None:
    try:
        links = await asyncio.to_thread(linker.generate_links, note_obj, embedding)
    except Exception:
        logger.warning("Link generation failed for note %s", note_obj.id)
```

The `handle_derive` function signature gains an optional `linker: Linker | None = None` parameter. The `create_worker()` factory also gains this parameter.

**Critical rules**: Agent responses are read for context but NEVER extracted from. User confirmations get `provenance: "user_confirmed"`. Notes stored with `is_buffered=1` for the Dreamer (Phase 5). Link generation failure NEVER blocks the write pipeline.

---

## Retrieval Pipeline (Phases 3–4)

No LLM calls in the read path. Pure local computation. Target: < 120ms end-to-end (Phase 4 budget; Phase 3 alone targets < 80ms).

### Full retrieval flow (Phase 4)

```
Query
  ├→ embed_query() ──→ Zvec search (top 30)
  ├→ FTS5 search (top 30)
  │         ↓
  │    Hydrate from SQLite + peer_id filter
  │         ↓
  │    ★ Link expansion: top 5 results → walk links → add neighbors to pool  [Phase 4]
  │         ↓
  │    RRF Fusion (k=60) — now includes link-expanded notes as third list
  │         ↓
  │    Score multipliers: decay × provenance × fatigue × inference_discount
  │         ↓
  │    Sort by composite score
  │         ↓
  │    ★ ColBERT rerank top 30 → keep top 10  [Phase 4, optional]
  │    OR MMR Dedup (cosine > 0.90 → drop lower-scored)  [Phase 3 fallback]
  │         ↓
  │    record_access() → return
```

Steps marked with ★ are Phase 4 additions. Without Phase 4 components, the pipeline falls back to Phase 3 behavior (no link expansion, MMR dedup instead of ColBERT).

### scorer.py — Pure scoring functions (Phase 3)

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

### fusion.py — RRF + MMR (Phase 3)

**`rrf_fuse(ranked_lists: list[list[str]], k=60) → dict[str, float]`**
- For each note across all lists: `score += 1 / (k + rank)` (0-indexed).
- Notes in multiple lists get multiple contributions.

**`mmr_dedup(scored_ids, embeddings: dict[str, list[float]], threshold=0.90) → list[str]`**
- Walk sorted IDs. Skip any candidate with cosine > threshold to an already-accepted result.
- nomic embeddings are L2-normalized, so cosine = dot product (use `np.dot`).
- Missing embeddings → auto-accept the note.

### retriever.py — Orchestrator (Phase 3, extended in Phase 4)

**`Retriever.__init__(db, zvec, embedder, colbert_reranker=None)`**

The `colbert_reranker` parameter is Phase 4. If `None`, retrieval uses Phase 3 behavior.

**`Retriever.retrieve(query, peer_id, limit=10) → list[RetrievalResult]`**

Steps:
1. **Parallel search**: FTS5 (async) overlapped with embed_query (sync ~10ms) + Zvec search (sync ~2-5ms).
2. **Hydrate**: collect all unique IDs → `db.get_notes_by_ids()` → filter by peer_id.
3. **Link expansion (Phase 4)**: Take the top `LINK_EXPANSION_TOP_SEEDS` (5) results by preliminary score. For each seed, call `db.get_linked_notes([seed_id], depth=1, max_per_seed=5)`. Deduplicate against already-known candidates. Add link-expanded notes to the pool with source `"link"`.
4. **RRF fusion**: build ranked ID lists (FTS, vector, and link-expanded if Phase 4) → `rrf_fuse()`. Track source ("fts"/"vector"/"link"/"both"/compound tags).
5. **Score**: `db.count_notes(peer_id)`. For each note compute days_since_access (from last_accessed_at or created_at), then call all scorer functions → composite score.
6. **Sort + final stage**:
   - **If ColBERT reranker available (Phase 4):** Sort by composite score → take top `COLBERT_RERANK_CANDIDATES` (30) → call `colbert_reranker.rerank(query, candidates, top_n=COLBERT_TOP_N)`. ColBERT score becomes the final score. If ColBERT fails, fall through to MMR.
   - **MMR dedup fallback (Phase 3):** sort by composite → batch-embed candidates via `embedder.embed_documents()` → `mmr_dedup()`. Cap on-the-fly embedding to ≤ 10 FTS-only notes.
7. **Truncate** to `limit`.
8. **Record access**: `db.record_access(final_ids)` — best-effort, never blocks.
9. **Return** `list[RetrievalResult]` sorted by score descending.

**Error handling**: Zvec fails → FTS-only. FTS fails → vector-only. Both fail → empty list. Link expansion fails → skip, use FTS+vector only. ColBERT fails → fall back to MMR dedup. record_access failure → log, still return.

---

## Intelligence Layer (Phase 4)

### intelligence/linker.py — A-MEM Link Generator

No LLM calls. Pure embedding similarity. Runs as part of the write pipeline (after note creation + embedding).

**`Linker.__init__(db: SQLiteStore, zvec: ZvecStore, embedder: Embedder)`** — Stores references.

**`Linker.generate_links(note: Note, embedding: list[float]) → list[Link]`**

For a newly created note, find candidate links and create them.

Steps:
1. **Find candidates**: Search Zvec with the note's embedding → top `LINK_MAX_CANDIDATES + 1` results (the +1 accounts for the note itself). Filter out the note's own ID.
2. **Filter by peer**: Only keep candidates with the same `peer_id` (hydrate via `db.get_notes_by_ids()`).
3. **Filter by threshold**: Only keep candidates with cosine similarity ≥ `LINK_SIMILARITY_THRESHOLD` (0.75). Zvec returns inner product scores for L2-normalized vectors, which equals cosine similarity. Use the score directly.
4. **Create links**: For each candidate above threshold, call `db.create_link()` with:
   - `link_type = "semantic"` (Phase 4 only creates semantic links; causal/temporal links are Phase 5 Dreamer)
   - `strength` = the cosine similarity score
   - Skip if a link already exists between the two notes (catch the UNIQUE constraint exception)
5. **Return** the list of created `Link` objects.

**Error handling**: If Zvec search fails, log a warning and return `[]`. If individual link creation fails (e.g., duplicate), log at DEBUG and skip. Never raise — linking failure must not block the write pipeline.

**`Linker.find_neighbors(note_id: str, max_results: int = 5) → list[tuple[Note, float]]`**

Public query method for retrieval. Fetches all links for `note_id` from `db.get_links()`, then hydrates the linked notes. Returns `(Note, link_strength)` tuples sorted by strength descending, capped at `max_results`.

### intelligence/reranker.py — ColBERT Reranker Wrapper

Wraps the `rerankers` library for ColBERT-based reranking of retrieval candidates.

**`ColBERTReranker.__init__(model_name: str = COLBERT_MODEL)`**

Load the model once at startup:

```python
from rerankers import Reranker

self._ranker = Reranker(model_name, model_type="colbert")
```

The model is ~130MB, 33M parameters, loads in ~2-3 seconds on CPU. Load once, reuse.

**`ColBERTReranker.rerank(query: str, candidates: list[tuple[str, str]], top_n: int = COLBERT_TOP_N) → list[tuple[str, float]]`**

Rerank candidate notes using ColBERT MaxSim scoring.

Parameters:
- `query`: The user's search query.
- `candidates`: List of `(note_id, note_content)` tuples.
- `top_n`: Return only the top N results.

Steps:
1. If `len(candidates) <= 1`, return candidates as-is.
2. Extract the list of document texts from candidates.
3. Call `self._ranker.rank(query=query, docs=doc_texts)`.
4. Map `RankedResults` back to note IDs using `.doc_id` (0-indexed position in input list).
5. Return `list[tuple[note_id, colbert_score]]` sorted by score descending, truncated to `top_n`.

**Error handling:** If reranking fails, log a warning and return the candidates in their original order.

**`ColBERTReranker.is_loaded() → bool`** — Check if the model loaded successfully.

### intelligence/profiler.py — Static Profile Generator

Generates a ~30-fact "Peer Card" from permanent notes via the existing Deriver API (DeepSeek V3.2). Called periodically or on-demand. Not user-facing.

**`Profiler.__init__(db: SQLiteStore, deriver: Deriver)`** — Stores references.

**`Profiler.generate(peer_id: str) → PeerProfile | None`**

Steps:
1. **Check minimum**: `db.get_permanent_notes(peer_id)` → if fewer than `PROFILE_MIN_NOTES`, return `None`.
2. **Collect input**: Take up to 50 permanent notes, sorted by importance DESC.
3. **Call Deriver API**: Use `deriver._call_api()` with the profile-generation system prompt and note texts as user content.
4. **Parse response**: Expect JSON with keys matching `PROFILE_SECTIONS`.
5. **Upsert**: `db.upsert_profile(peer_id, sections, fact_count, source_note_ids)`.
6. **Return** the `PeerProfile`.

**Profile generation system prompt:**

```python
PROFILE_SYSTEM_PROMPT = """\
You generate a concise static profile (Peer Card) from a collection of permanent facts \
about a person.

Organize the facts into exactly these sections:
- identity: name, age, location, key personal identifiers
- professional: role, company, industry (do NOT include skill level or expertise — \
those belong in dynamic memory)
- communication_style: preferred tone, formality, response length preferences
- relationships: important people, pets, family mentions

Rules:
- Maximum 30 facts total across all sections.
- Each fact is one short sentence.
- Omit sections with no relevant facts (return empty string for that section).
- Prioritize facts with higher importance scores.
- Do NOT infer or hallucinate — only include facts directly supported by the input notes.

Return JSON: {"identity": "...", "professional": "...", "communication_style": "...", \
"relationships": "..."}
"""
```

**Error handling:** If the API call fails, log a warning and return `None`.

**`Profiler.get_profile_text(peer_id: str) → str | None`**

Fetch the profile from SQLite and format as text for system prompt injection:

```
## About {peer_name}
### Identity
{identity facts}
### Professional
{professional facts}
### Communication Style
{communication_style facts}
### Relationships
{relationships facts}
```

Returns `None` if no profile exists.

### intelligence/__init__.py

```python
from mnemosyne.intelligence.linker import Linker
from mnemosyne.intelligence.reranker import ColBERTReranker
from mnemosyne.intelligence.profiler import Profiler


def create_linker(db, zvec, embedder) -> Linker:
    return Linker(db=db, zvec=zvec, embedder=embedder)


def create_reranker() -> ColBERTReranker:
    return ColBERTReranker()


def create_profiler(db, deriver) -> Profiler:
    return Profiler(db=db, deriver=deriver)


__all__ = [
    "ColBERTReranker", "Linker", "Profiler",
    "create_linker", "create_profiler", "create_reranker",
]
```

---

## Latency Targets

| Path | Target | Notes |
|------|--------|-------|
| Fast path (write + enqueue) | < 5ms | User never waits past this |
| Full async write pipeline | < 5s | Not user-facing |
| Link generation (write path) | < 50ms | Zvec search + SQLite inserts. Not user-facing. |
| FTS5 search | < 1ms | SQLite in-process |
| Zvec vector search | ~2-5ms | HNSW in-process |
| Query embedding | ~10ms | Single call |
| Link expansion | < 5ms | SQLite queries only |
| RRF + scoring | < 1ms | Pure math, ~60 candidates |
| MMR dedup | ~20-50ms | Batch embed + dot products (Phase 3 fallback) |
| ColBERT reranking (30 candidates) | < 30ms | CPU, short texts (~20-50 words each) |
| Profile generation | < 5s | LLM call, not user-facing. Background task. |
| **Full retrieval (Phase 3, no ColBERT)** | **< 80ms** | **User-facing** |
| **Full retrieval (Phase 4, with ColBERT)** | **< 120ms** | **User-facing** |

---

## Environment Variables

```bash
export NOUSRESEARCH_API_KEY="your-key"   # Required for Phase 2 Deriver + Phase 4 Profiler
```

No new env vars for Phases 3-4. The ColBERT model (`answerdotai/answerai-colbert-small-v1`) downloads automatically on first load (~130MB from HuggingFace).

---

## Critical Library Notes

### rerankers (v0.10.0+) — Phase 4

```python
from rerankers import Reranker

# Load ColBERT model (one-time, ~2-3s on CPU, ~130MB)
ranker = Reranker("answerdotai/answerai-colbert-small-v1", model_type="colbert")

# Rerank documents
results = ranker.rank(query="What pets does the user have?", docs=["User has a cat", "User likes hiking"])

# Access results
for r in results.results:
    print(r.doc_id, r.score, r.text)
# r.doc_id is the 0-indexed position in the original docs list
```

- `model_type="colbert"` is REQUIRED. Without it, rerankers may misidentify the model.
- The `rank()` method returns a `RankedResults` object. Access `.results` for the list of `Result` objects.
- Each `Result` has: `.doc_id` (int, 0-indexed position), `.score` (float), `.text` (str).
- CPU inference is fine for 20-50 short text candidates (~20ms).
- Install with: `pip install "rerankers[transformers]"` (needs the transformers extra for local models).

### Zvec score semantics

Zvec's `query()` returns results with a `.score` field. For HNSW with L2-normalized vectors (which nomic-embed produces), the score represents inner product similarity (higher = more similar). Since our embeddings are L2-normalized, inner product = cosine similarity. Use the score directly for the link threshold comparison in the Linker.

---

## Tests

All use `tmp_path`. All API calls mocked. Expensive models (embedder, ColBERT) use `scope="module"` fixtures.

### Phase 1
- **test_sqlite_store.py**: schema, CRUD, FTS5 triggers, task queue atomic dequeue, dead-letter.
- **test_models.py**: Pydantic model construction, from_row, JSON parsing, defaults.
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

### Phase 4
- **test_linker.py**:
  - test_creates_semantic_links: Insert 3 similar notes → linker creates links between them with strength ≥ threshold.
  - test_no_links_below_threshold: Insert 2 unrelated notes → linker creates no links.
  - test_self_link_excluded: The note itself does not appear in link candidates.
  - test_peer_isolation: Notes from different peers do not get linked.
  - test_duplicate_link_skipped: Running linker twice on the same note does not raise or create duplicate links.
  - test_zvec_failure_graceful: Zvec search failure → returns [], no crash.
  - test_find_neighbors: Create notes with links → find_neighbors() returns linked notes sorted by strength.

- **test_reranker.py**:
  - test_reranks_candidates: Given a query and 5 candidates, reranker returns them reordered by ColBERT score.
  - test_single_candidate_passthrough: 1 candidate → returned as-is.
  - test_empty_candidates: 0 candidates → returns [].
  - test_top_n_truncation: 10 candidates, top_n=3 → returns exactly 3.
  - test_scores_are_floats: All returned scores are float.
  - test_relevant_ranked_higher: Query "cats and dogs" with candidates including one about pets → pet candidate ranked in top 3.

- **test_profiler.py**:
  - test_generates_profile: Mock Deriver → profile has all 4 sections, stored in SQLite.
  - test_skips_below_min_notes: Peer with < 5 permanent notes → returns None.
  - test_profile_text_format: Generated text includes section headers and facts.
  - test_updates_existing_profile: Second generation replaces the first.
  - test_api_failure_returns_none: Deriver fails → returns None, no crash.

- **test_link_expansion.py**:
  - test_linked_notes_added_to_results: Create 3 linked notes → retrieval for one returns linked neighbors.
  - test_expansion_respects_depth: Depth=1 → only direct neighbors, not neighbors-of-neighbors.
  - test_expansion_respects_max: More than LINK_EXPANSION_MAX neighbors → only top N by strength included.
  - test_no_links_no_expansion: Note with no links → retrieval results unchanged.
  - test_link_source_tag: Link-expanded notes have source containing "link".

- **test_intelligence_pipeline.py**:
  - test_full_write_with_links: Ingest message → derive → notes created → links generated between similar notes.
  - test_retrieval_with_colbert: Add notes → retrieve with ColBERT reranker → results have colbert_score set.
  - test_retrieval_colbert_fallback: ColBERT unavailable → retrieval still works (Phase 3 behavior).
  - test_profile_generation_round_trip: Create peer → add permanent notes → generate profile → retrieve profile text.
  - test_link_expansion_in_retrieval: Create linked notes → retrieve → linked notes appear in results with "link" source.

---

## Design Principles

1. **User never waits for memory processing.** Write path is async; read path is < 120ms (< 80ms without ColBERT).
2. **Raw data is immutable.** Stream 1 observations are ground truth.
3. **Frequency ≠ importance.** Two-dimensional scoring prevents topic domination.
4. **Decay scores, never deletes.** All data preserved; relevance is temporal.
5. **Agent responses are context, not memory.** Read but never stored as user attributes.
6. **Graceful degradation everywhere.** If one component fails, others still return results. Zvec down → no links. ColBERT down → composite scoring with MMR. Deriver down → no profile. Link generation down → write pipeline continues.
7. **No LLM calls in the read path** (Phases 3–4). ColBERT and link expansion are pure local computation. Profile generation is a background task.
8. **Semantic links only in Phase 4.** Causal, temporal, and contradiction links are Phase 5 (Dreamer). The link_type field and infrastructure exist, but Phase 4 only creates "semantic" links.
9. **Profile excludes expertise.** Per the architecture spec, expertise/skill level is NOT in the static profile. It stays in dynamic memory to prevent model over-anchoring.
10. **ColBERT is optional.** The retriever works without ColBERT (falls back to Phase 3 MMR dedup). This allows deployment without the extra model.
