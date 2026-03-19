# Mnemosyne Phase 1 — Foundation Spec

Phase 1 gets data flowing: SQLite schema, FTS5 search, embedding model, Zvec vector store, and a Python API tying them together. No LLM calls — that's Phase 2.

---

## Project Structure

```
mnemosyne/
├── pyproject.toml
├── CLAUDE.md
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
│   └── utils/
│       ├── __init__.py
│       └── ids.py
└── tests/
    ├── conftest.py
    ├── test_sqlite_store.py
    ├── test_zvec_store.py
    ├── test_embedder.py
    └── test_memory_api.py
```

## Dependencies (pyproject.toml)

```toml
[project]
name = "mnemosyne"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "aiosqlite>=0.20.0",
    "zvec>=0.2.0",
    "sentence-transformers>=5.0.0",
    "onnxruntime>=1.18.0",
    "pydantic>=2.0.0",
    "python-ulid>=3.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]
```

No other dependencies. Do NOT add fastapi, uvicorn, httpx, ragatouille, or networkx.

---

## config.py

```python
DEFAULT_DATA_DIR = "./mnemosyne_data"
SQLITE_DB_NAME = "mnemosyne.db"
ZVEC_COLLECTION_DIR = "zvec_notes"
ZVEC_COLLECTION_NAME = "mnemosyne_notes"

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 384
EMBEDDING_QUERY_PREFIX = "search_query: "
EMBEDDING_DOC_PREFIX = "search_document: "

RRF_K = 60
SCHEMA_VERSION = 1

PROVENANCE_ORGANIC = "organic"
PROVENANCE_AGENT_PROMPTED = "agent_prompted"
PROVENANCE_USER_CONFIRMED = "user_confirmed"
PROVENANCE_INFERRED = "inferred"

NOTE_TYPE_OBSERVATION = "observation"
NOTE_TYPE_INFERENCE = "inference"

DURABILITY_PERMANENT = "permanent"
DURABILITY_CONTEXTUAL = "contextual"
DURABILITY_EPHEMERAL = "ephemeral"

LINK_TYPES = ["semantic", "causal", "temporal", "contradicts", "supports", "derived_from"]

DEFAULT_CONFIDENCE_OBSERVATION = 0.8
DEFAULT_CONFIDENCE_INFERENCE = 0.6
DEFAULT_CONFIDENCE_USER_SET = 1.0
DEFAULT_EMOTIONAL_WEIGHT = 0.5
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

**notes** — the core memory unit. All columns listed, include ALL even if unused in Phase 1:
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

**task_queue**: id (PK), task_type, payload (JSON), status ('pending'|'processing'|'completed'|'failed' default 'pending'), priority (INT default 0), attempts (INT default 0), max_attempts (INT default 3), error (nullable), created_at, started_at (nullable), completed_at (nullable). Index: (status, priority DESC).

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
- **Task queue**: enqueue_task, dequeue_task(task_type) — atomic claim via UPDATE...RETURNING, complete_task, fail_task
- **FTS5**: fts_search(query, peer_id, limit) — match on notes_fts, join to notes, filter by peer_id, return with BM25 scores

---

## Embedder

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

## ZvecStore

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

## MemoryAPI

Coordinates SQLite + Zvec + Embedder. This is the public interface.

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

## Tests

All use `tmp_path` fixture for isolation.

**test_sqlite_store.py**: Schema creates all tables. CRUD works. FTS5 finds notes. FTS5 triggers sync correctly (insert→findable, delete→gone, update→reflects new content). Task queue atomic dequeue works.

**test_embedder.py**: Model loads. Returns exactly 384-dim. Different texts→different embeddings. Same text→identical. Doc vs query prefix→different embeddings for same raw text.

**test_zvec_store.py**: Insert+query finds the doc. Batch insert returns ranked results. Delete removes from results. Empty query returns [].

**test_memory_api.py**: Full round trip: create peer → start session → add message → add note → keyword search finds it → vector search finds it → hybrid search finds it. Notes exist in both SQLite and Zvec independently.
