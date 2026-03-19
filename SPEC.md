# Mnemosyne — Complete Spec (Phases 1–2)

This is the implementation spec for Mnemosyne, a next-generation AI agent memory system. Phase 1 builds the foundation: SQLite schema, FTS5 search, embedding model, Zvec vector store, and a coordinating API. Phase 2 adds the async write pipeline: fast-path message intake, the two-stage Deriver (Extractor + Scorer) calling DeepSeek V3.2 via NousResearch, embedding, and multi-store writes.

---

## Project Structure

```
mnemosyne/
├── pyproject.toml
├── CLAUDE.md
├── SPEC.md                     ← this file
├── mnemosyne/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── db/
│   │   ├── __init__.py
│   │   └── sqlite_store.py
│   ├── vectors/
│   │   ├── __init__.py
│   │   ├── zvec_store.py
│   │   └── embedder.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── memory_api.py
│   ├── pipeline/               # Phase 2: write pipeline
│   │   ├── __init__.py
│   │   ├── intake.py           # Fast-path message ingestion
│   │   ├── deriver.py          # Extractor + Scorer (DeepSeek V3.2)
│   │   ├── handlers.py         # Task handlers (handle_derive)
│   │   └── worker.py           # Queue worker loop + dispatch
│   └── utils/
│       ├── __init__.py
│       └── ids.py
└── tests/
    ├── conftest.py
    ├── test_sqlite_store.py
    ├── test_zvec_store.py
    ├── test_embedder.py
    ├── test_memory_api.py
    ├── test_worker.py           # Phase 2
    ├── test_intake.py           # Phase 2
    ├── test_deriver.py          # Phase 2
    └── test_write_pipeline.py   # Phase 2 (integration)
```

---

## Dependencies (pyproject.toml)

```toml
[project]
name = "mnemosyne"
version = "0.2.0"
requires-python = ">=3.11"
dependencies = [
    "aiosqlite>=0.20.0",
    "zvec>=0.2.0",
    "sentence-transformers>=5.0.0",
    "onnxruntime>=1.18.0",
    "pydantic>=2.0.0",
    "python-ulid>=3.0.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]
```

Do NOT add fastapi, uvicorn, ragatouille, or networkx. Those are later phases.

---

## config.py

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

# Retrieval
RRF_K = 60

# Schema
SCHEMA_VERSION = 1

# Provenance
PROVENANCE_ORGANIC = "organic"
PROVENANCE_AGENT_PROMPTED = "agent_prompted"
PROVENANCE_USER_CONFIRMED = "user_confirmed"
PROVENANCE_INFERRED = "inferred"

# Note types
NOTE_TYPE_OBSERVATION = "observation"
NOTE_TYPE_INFERENCE = "inference"

# Durability
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

# Deriver API (Phase 2)
NOUSRESEARCH_BASE_URL = "https://portal.nousresearch.com/api-docs"
NOUSRESEARCH_MODEL = "deepseek/deepseek-v3.2"
DERIVER_EXTRACT_TEMPERATURE = 0.1
DERIVER_SCORE_TEMPERATURE = 0.1
DERIVER_MAX_RETRIES = 3
DERIVER_RETRY_DELAYS = [1.0, 2.0, 4.0]
WORKER_POLL_INTERVAL = 2.0


```

---

## SQLite Schema

### PRAGMAs (apply on every connection open)

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -8000;
PRAGMA busy_timeout = 5000;
PRAGMA mmap_size = 268435456;
PRAGMA temp_store = MEMORY;
PRAGMA foreign_keys = ON;
```

### Tables

All PKs are TEXT (ULIDs). All timestamps: `strftime('%Y-%m-%dT%H:%M:%fZ', 'now')`.

**config**: key (PK), value, updated_at. Insert `('schema_version', '1')` on init.

**peers**: id (PK), name, peer_type ('user'|'agent' default 'user'), static_profile (JSON nullable), profile_updated_at (nullable), created_at, metadata (JSON default '{}').

**sessions**: id (PK), peer_id (FK→peers), started_at, ended_at (nullable), summary (nullable), metadata (JSON). Indexes: peer_id, started_at.

**messages**: id (PK), session_id (FK→sessions), peer_id (FK→peers), role ('user'|'assistant'|'system'), content, created_at, metadata (JSON). Indexes: session_id, created_at.

**notes** — the core memory unit:
- id (PK), peer_id (FK), session_id (FK nullable), source_message_id (FK nullable)
- content (TEXT NOT NULL), context_description (nullable), keywords (JSON array default '[]'), tags (JSON array default '[]')
- note_type ('observation'|'inference' default 'observation'), provenance (default 'organic'), durability (default 'contextual')
- emotional_weight (REAL default 0.5), importance (REAL default 0.0), confidence (REAL default 0.8)
- evidence_count (INT default 1), unique_sessions_mentioned (INT default 1)
- q_value (REAL default 0.0), access_count (INT default 0), last_accessed_at (nullable), times_surfaced (INT default 0), decay_score (REAL default 1.0)
- is_buffered (INT default 1), canonical_note_id (nullable)
- created_at, updated_at
- zvec_id (nullable)
- Indexes: peer_id, session_id, note_type, durability, is_buffered, decay_score, importance, created_at

**links**: id (PK), source_note_id (FK→notes), target_note_id (FK→notes), link_type, strength (REAL default 0.5), created_at, metadata (JSON). UNIQUE(source_note_id, target_note_id, link_type). Indexes: source, target, type.

**task_queue**: id (PK), task_type, payload (JSON), status ('pending'|'processing'|'completed'|'failed'|'dead_letter' default 'pending'), priority (INT default 0), attempts (INT default 0), max_attempts (INT default 3), error (nullable), created_at, started_at (nullable), completed_at (nullable). Index: (status, priority DESC).

### FTS5

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    content, context_description, keywords, tags,
    content='notes', content_rowid='rowid'
);
```

REQUIRED sync triggers (without these, FTS5 returns nothing):

```sql
-- After INSERT: copy new row into FTS
CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(rowid, content, context_description, keywords, tags)
    VALUES (new.rowid, new.content, new.context_description, new.keywords, new.tags);
END;

-- After DELETE: remove from FTS
CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, content, context_description, keywords, tags)
    VALUES ('delete', old.rowid, old.content, old.context_description, old.keywords, old.tags);
END;

-- After UPDATE: delete old + insert new
CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, content, context_description, keywords, tags)
    VALUES ('delete', old.rowid, old.content, old.context_description, old.keywords, old.tags);
    INSERT INTO notes_fts(rowid, content, context_description, keywords, tags)
    VALUES (new.rowid, new.content, new.context_description, new.keywords, new.tags);
END;
```

### SQLiteStore CRUD Methods

All async. Parameterized queries only.

- **Peers**: create_peer, get_peer, update_peer, list_peers
- **Sessions**: create_session, get_session, end_session, list_sessions(peer_id)
- **Messages**: add_message, get_messages(session_id, limit), get_recent_context(session_id, n_turns)
- **Notes**: create_note, get_note, update_note, delete_note, list_notes(peer_id, filters), get_buffered_notes(peer_id)
- **Links**: create_link, get_links(note_id), delete_link
- **Task queue**: enqueue_task, dequeue_task(task_type) — atomic claim via UPDATE...RETURNING, complete_task, fail_task (with dead_letter transition when attempts >= max_attempts)
- **FTS5**: fts_search(query, peer_id, limit) — match on notes_fts, join to notes, filter by peer_id, return with BM25 scores

---

## Embedder (mnemosyne/vectors/embedder.py)

Wraps nomic-embed-text-v1.5. Handles prefix prepending internally — callers pass raw text. Uses ONNX backend for ~3x CPU speedup, with PyTorch fallback.

```python
class Embedder:
    def __init__(self):
        """
        Load model with trust_remote_code=True, truncate_dim=384.
        Try backend="onnx" first (requires onnxruntime). If ONNX fails
        for any reason (missing dep, export error, etc.), fall back to
        PyTorch backend. Log which backend loaded.
        """

    @property
    def dimension(self) -> int: return 384

    @property
    def backend(self) -> str:
        """Return 'onnx' or 'torch' depending on which loaded."""

    def embed_document(self, text: str) -> list[float]:
        """Prepends 'search_document: ' then encodes."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch version."""

    def embed_query(self, query: str) -> list[float]:
        """Prepends 'search_query: ' then encodes."""
```

Convert numpy arrays to `list[float]` for Zvec. Embeddings are L2-normalized by default.

---

## ZvecStore (mnemosyne/vectors/zvec_store.py)

```python
class ZvecStore:
    def __init__(self, data_dir: str):
        """Open/create 384-dim FP32 collection."""

    def insert(self, note_id: str, embedding: list[float]) -> None
    def insert_batch(self, items: list[tuple[str, list[float]]]) -> None
    def search(self, query_embedding: list[float], top_k: int = 20) -> list[dict]
    def delete(self, note_id: str) -> None
    def optimize(self) -> None
    def stats(self) -> dict
```

Wrap all calls in try/except. Do not let Zvec errors crash the system.

---

## MemoryAPI (mnemosyne/api/memory_api.py)

Coordinates SQLite + Zvec + Embedder. This is the public interface for direct operations.

```python
class MemoryAPI:
    async def initialize(self) -> None
    async def close(self) -> None

    async def create_peer(name, peer_type="user") -> Peer
    async def get_peer(peer_id) -> Peer | None

    async def start_session(peer_id) -> Session
    async def end_session(session_id) -> Session

    async def add_message(session_id, peer_id, role, content) -> Message

    async def add_note(peer_id, content, session_id=None, note_type="observation",
                       provenance="organic", durability="contextual",
                       emotional_weight=0.5, keywords=None, tags=None,
                       context_description=None) -> Note

    async def get_note(note_id) -> Note | None

    async def search_keyword(query, peer_id, limit=20) -> list[Note]
    async def search_vector(query, peer_id, top_k=20) -> list[tuple[Note, float]]
    async def search_hybrid(query, peer_id, limit=10) -> list[tuple[Note, float]]
```

**add_note** flow: generate ULID → embed content → insert SQLite → insert Zvec → optimize → return Note.

**search_vector** flow: embed query → Zvec search → fetch full notes from SQLite → filter by peer_id → return (Note, score) tuples.

**search_hybrid** uses RRF: run keyword + vector in parallel, combine with `rrf_score = Σ 1/(60 + rank_i)`, sort descending, return top limit.

---

## Models (Pydantic)

Define: Peer, Session, Message, Note, Link, TaskItem. Each with all columns from its table. Include `from_row()` classmethods to construct from aiosqlite Row dicts.

---

## Write Pipeline (Phase 2)

The user never waits for memory processing. Everything after the fast-path SQLite write is async.

### Intake (mnemosyne/pipeline/intake.py)

Fast-path message ingestion. Synchronous write + enqueue, < 5ms target.

```python
async def ingest_message(
    session_id: str,
    peer_id: str,
    role: str,
    content: str,
    db: SQLiteStore,
) -> str:
    """
    1. Write message to messages table via db.add_message()
    2. If role == "user":
       a. Fetch preceding context via db.get_recent_context(session_id, n_turns=3)
       b. Enqueue task via db.enqueue_task(
            task_type="derive",
            payload={
                "message_id": message_id,
                "session_id": session_id,
                "peer_id": peer_id,
                "content": content,
                "preceding_turns": [...],  # list of {role, content} dicts
            }
          )
    3. Return message_id immediately.

    Assistant messages are stored but do NOT trigger derivation.
    """
```

### Deriver (mnemosyne/pipeline/deriver.py)

Two-stage LLM pipeline calling DeepSeek V3.2 via NousResearch.

```python
class Deriver:
    def __init__(
        self,
        api_key: str,
        base_url: str = NOUSRESEARCH_BASE_URL,
        model: str = NOUSRESEARCH_MODEL,
    ):
        """Create httpx.AsyncClient. Store config."""

    async def extract(
        self,
        user_message: str,
        preceding_turns: list[dict],
    ) -> list[dict]:
        """
        Deriver Call 1 — Extractor.

        System prompt instructs the model to:
        - Extract atomic fact notes from the user's message ONLY
        - Read preceding turns for context but NEVER extract from assistant messages
        - If user confirms something agent said, mark is_confirmation=true
        - Return JSON: {"notes": [{"text": str, "is_confirmation": bool}, ...]}

        API call:
        - POST {base_url}/chat/completions
        - Headers: Content-Type, Authorization: Bearer {api_key}
        - Body: model, messages, temperature=0.1, response_format={"type": "json_object"}

        Retry: 3 attempts with 1s/2s/4s backoff on 429 and 5xx.
        On persistent parse failure: return empty list, log error.

        Returns: list of {"text": str, "is_confirmation": bool}
        """

    async def score(self, notes: list[dict]) -> list[dict]:
        """
        Deriver Call 2 — Scorer.

        System prompt instructs the model to tag each note with:
        - emotional_weight (0.0-1.0)
        - provenance: "organic"|"agent_prompted"|"user_confirmed"|"inferred"
          (pass through "user_confirmed" if is_confirmation was true)
        - durability: "permanent"|"contextual"|"ephemeral"
        - keywords: list[str]
        - tags: list[str]
        - context_description: str

        Returns JSON: {"scored_notes": [{all original fields + metadata}, ...]}

        Same retry/backoff logic as extract().
        Returns: list of fully-scored note dicts.
        """

    async def close(self) -> None:
        """Close httpx client."""
```

**Critical rules for Deriver prompts:**
- Agent responses are READ for context but NEVER extracted from
- Exception: user confirmation ("yeah that's right") → provenance: "user_confirmed"
- Each note must be atomic — one fact per note
- Empty extraction is valid (some messages contain no memorable facts)

### Handler (mnemosyne/pipeline/handlers.py)

Wires the full derive pipeline together.

```python
async def handle_derive(
    task: TaskItem,
    db: SQLiteStore,
    deriver: Deriver,
    embedder: Embedder,
    zvec: ZvecStore,
) -> None:
    """
    Full derive handler, called by the Worker when task_type="derive".

    Steps:
    1. Parse payload: message_id, session_id, peer_id, content, preceding_turns
    2. Extract: notes = await deriver.extract(content, preceding_turns)
       - If empty, return early (some messages have no extractable facts)
    3. Score: scored_notes = await deriver.score(notes)
    4. For each scored note:
       a. Embed: embedding = embedder.embed_document(note["text"])
       b. Store in SQLite via db.create_note(
            peer_id, content, session_id, source_message_id,
            note_type="observation", provenance, durability, emotional_weight,
            keywords, tags, context_description,
            is_buffered=1,  ← marks as waiting for Dreamer
          )
       c. Store embedding in Zvec via zvec.insert(note_id, embedding)
          - If Zvec fails: log error, do NOT raise. SQLite write already succeeded.
            Leave zvec_id as NULL for later retry.
       d. If Zvec succeeded: await db.update_note(note_id, zvec_id=note_id)
    5. Log: number of notes extracted, scored, stored + per-stage latency.
    """
```

**Important:** Notes go into the `notes` table with `is_buffered=1`. There is NO separate `raw_notes` table. The `is_buffered` flag is what the Dreamer (Phase 5) uses to find unprocessed notes.

### Worker (mnemosyne/pipeline/worker.py)

Polls SQLiteStore for pending tasks and dispatches to registered handlers.

```python
class Worker:
    def __init__(self, db: SQLiteStore, handler_map: dict[str, Callable]):
        """
        db: SQLiteStore instance (has enqueue/dequeue/complete/fail methods)
        handler_map: {"derive": handle_derive, ...}
        """

    async def run(self, poll_interval: float = WORKER_POLL_INTERVAL) -> None:
        """
        Loop forever:
        1. Call db.dequeue_task() — atomic claim via UPDATE...RETURNING
        2. If no task, sleep poll_interval and continue
        3. Look up handler from handler_map[task.task_type]
        4. Call handler(task, ...) in try/except
        5. On success: db.complete_task(task.id)
        6. On failure: if task.attempts < task.max_attempts → db.fail_task()
           resets status to 'pending' and increments attempts.
           If attempts >= max_attempts → set status to 'dead_letter'.
        """

    async def run_once(self) -> bool:
        """Process one task and return. Returns True if a task was processed.
        Used for testing — avoids infinite loop."""
```

The Worker does NOT own any SQL. All queue CRUD lives on SQLiteStore. The Worker only contains the polling loop and dispatch logic.

### Factory (mnemosyne/pipeline/__init__.py)

```python
def create_worker(db: SQLiteStore, deriver: Deriver, embedder: Embedder, zvec: ZvecStore) -> Worker:
    """Wraps handle_derive with all dependencies injected, returns a Worker."""
```

---

## API Configuration (Phase 2)

| Setting | Value |
|---------|-------|
| Provider | NousResearch |
| Base URL | `https://inference-api.nousresearch.com/v1` |
| Model | `deepseek/deepseek-v3.2` |
| Auth | `Authorization: Bearer $NOUSRESEARCH_API_KEY` |
| HTTP client | `httpx.AsyncClient` (never openai SDK) |
| Temperature | 0.1 for both Extractor and Scorer |
| Response format | `{"type": "json_object"}` |
| Retry | 3 attempts, 1s/2s/4s backoff on 429 + 5xx |

---

## Latency Targets

| Path | Target | Notes |
|------|--------|-------|
| Fast path (write + enqueue) | < 5ms | User never waits past this |
| Extractor API call | ~1-3s | Async, fire-and-forget |
| Scorer API call | ~0.5-1s | Async, shorter input |
| Embedding (per note) | ~10ms | CPU, nomic-embed-text-v1.5 |
| Zvec insert (per note) | < 1ms | In-process |
| Full async pipeline | < 5s total | Not user-facing |

---

## Environment Variables

```bash
# Required for Phase 2
export NOUSRESEARCH_API_KEY="your-nousresearch-API-key"
```

---

## Tests

All use `tmp_path` fixture for isolation. All DeepSeek API calls are mocked — never make real API calls in the test suite.

### Phase 1 Tests

**test_sqlite_store.py**: Schema creates all tables. CRUD works. FTS5 finds notes. FTS5 triggers sync correctly (insert→findable, delete→gone, update→reflects new content). Task queue atomic dequeue works. Dead-letter transition after max_attempts.

**test_embedder.py**: Model loads. Returns exactly 384-dim. Different texts→different embeddings. Same text→identical. Doc vs query prefix→different embeddings for same raw text.

**test_zvec_store.py**: Insert+query finds the doc. Batch insert returns ranked results. Delete removes from results. Empty query returns [].

**test_memory_api.py**: Full round trip: create peer → start session → add message → add note → keyword search finds it → vector search finds it → hybrid search finds it. Notes exist in both SQLite and Zvec independently.

### Phase 2 Tests

**test_worker.py**: Worker.run_once() processes a pending task and calls the handler. Returns False when queue is empty. Catches handler exceptions and calls fail_task. Dead-letter after max_attempts. Atomic dequeue prevents double-processing.

**test_intake.py**: User message writes to SQLite and enqueues a derive task. Assistant message writes but does NOT enqueue. Preceding context correctly gathered (up to 3 turns). Message findable via FTS5 after ingestion.

**test_deriver.py**: Extraction produces atomic notes (mocked API). Confirmation detection works. Empty message → empty list. Scorer returns all required fields with correct types. Scorer passes through user_confirmed provenance. Retry on 429. Graceful failure on garbage JSON. Scorer with empty input → no API call.

**test_write_pipeline.py** (integration): Full round trip — ingest_message → worker processes task → notes exist in SQLite with all metadata → notes findable via FTS5 → notes findable via Zvec → task status is "completed". Pipeline survives API failure + retry. Empty extraction completes gracefully. Assistant messages produce no derived notes.

---

## Design Principles

1. **User never waits for memory processing.** Everything after the SQLite write is async.
2. **Raw data is immutable.** Notes in Stream 1 (is_buffered or not) are ground truth.
3. **Frequency ≠ importance.** Two-dimensional scoring (emotional weight × frequency) prevents topic domination.
4. **Memory informs, doesn't dominate.** Anti-sycophancy guards prevent self-reinforcing loops.
5. **Decay scores, never deletes.** All data is preserved; relevance is temporal.
6. **Expertise stays dynamic.** Not in the static profile to prevent model over-anchoring.
7. **Agent responses are context, not memory.** Read but never stored as the user's attributes.
8. **Ephemeral things fade naturally.** Durability classification prevents task noise from polluting long-term memory.
9. **Graceful degradation.** If Zvec fails, SQLite still has the data. If the API fails, retry. If retry fails, dead-letter for inspection.
