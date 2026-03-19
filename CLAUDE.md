# Mnemosyne — AI Agent Memory System

## Overview
Mnemosyne is a next-generation AI agent memory system using SQLite + Zvec (embedded vector DB) + nomic-embed-text-v1.5. No external servers — everything runs in-process.

## Stack
Python 3.11. SQLite (aiosqlite) for structured data + FTS5. Zvec v0.2.0 (Alibaba) for vector search. sentence-transformers + ONNX Runtime for embeddings. Pydantic v2 for models. pytest + pytest-asyncio for tests.

## Target Deployment
Hostinger VPS: 4 vCPU, 16GB RAM, Ubuntu 24.04, Docker. CPU-only — no GPU.

## Key Directories
- `mnemosyne/` — main package
- `mnemosyne/db/` — SQLite store, schema, CRUD
- `mnemosyne/vectors/` — Zvec store + embedding model wrapper
- `mnemosyne/api/` — high-level MemoryAPI coordinating SQLite + Zvec
- `mnemosyne/models.py` — Pydantic data models
- `mnemosyne/config.py` — all constants and configuration
- `tests/` — pytest test suite

## Commands
- Use `python3` (not `python`) to run all commands.
- Install: `python3 -m pip install -e ".[dev]"`
- Test: `python3 -m pytest tests/ -v`
- Test single file: `python3 -m pytest tests/test_sqlite_store.py -v`

## Code Standards
- Type hints on every function signature and return type. Use `str | None` not `Optional[str]`.
- Docstrings on every public method.
- Async all SQLite operations via aiosqlite. Zvec is sync — wrap with `asyncio.to_thread()`.
- Parameterized SQL queries only. Never f-strings for SQL.
- All IDs are ULIDs (26-char, time-sortable strings). Use `python-ulid`: `from ulid import ULID; str(ULID())`.
- Log with `logging.getLogger(__name__)`. INFO for operations, DEBUG for details, WARNING for fallbacks.

## Critical Library Notes

### nomic-embed-text-v1.5

The model repo on HuggingFace already contains pre-exported ONNX weights in its `onnx/` directory. sentence-transformers can load these directly with `backend="onnx"` for ~3x CPU speedup. Use ONNX as the primary backend, with PyTorch as fallback.

```python
from sentence_transformers import SentenceTransformer

# Primary: ONNX backend (~3x faster on CPU)
try:
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        truncate_dim=384,
        backend="onnx",
    )
except Exception:
    # Fallback: PyTorch backend
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        truncate_dim=384,
    )
```
- `trust_remote_code=True` is REQUIRED for both backends. Without it the model fails to load.
- `truncate_dim=384` activates Matryoshka truncation (768 → 384 dims).
- REQUIRES task prefixes: prepend `"search_document: "` for documents, `"search_query: "` for queries.
- ONNX backend requires `onnxruntime` package. If it's not installed, fall back to PyTorch.
- Model is ~500MB, downloads on first use. The ONNX variant may download additional files.

### Zvec (v0.2.0)
```python
import zvec
schema = zvec.CollectionSchema(
    name="mnemosyne_notes",
    vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, 384),
)
collection = zvec.create_and_open(path="./data/zvec_notes", schema=schema)
collection.insert([zvec.Doc(id="note_abc", vectors={"embedding": [0.1, ...]})])
collection.optimize()  # MUST call after inserts to build HNSW index
results = collection.query(zvec.VectorQuery("embedding", vector=[0.1, ...]), topk=20)
collection.delete(ids="note_abc")
```
- Synchronous API — use `asyncio.to_thread()` from async code.
- Do NOT use scalar FieldSchema fields — the API is too new and may be flaky. Filter by peer_id on the SQLite side.
- `collection.optimize()` MUST be called after inserts.

### python-ulid
```python
from ulid import ULID
new_id = str(ULID())  # e.g. "01HASFKBN8SKZTSVVS03K5AMMS"
```

### aiosqlite
```python
async with aiosqlite.connect(path) as db:
    db.row_factory = aiosqlite.Row  # dict-like access
    await db.execute("PRAGMA journal_mode = WAL")
```

## Known Gotchas
- FTS5 virtual tables with `content=` (external content) REQUIRE manual sync triggers (AFTER INSERT, AFTER DELETE, AFTER UPDATE on the source table). Without these triggers, FTS5 search will not find any data.
- Zvec `collection.optimize()` must be called after inserts or the HNSW index won't be built and queries return nothing.
- nomic-embed produces different embeddings for the same text with different prefixes. Document and query prefixes are NOT interchangeable.
- The ONNX backend failing to load is handled by fallback — it is not an error. Log it at WARNING and continue with PyTorch. Both backends produce identical embeddings.

## Verification
After any code change:
1. `pytest tests/ -v` — all tests pass
2. Manually verify: add a note → keyword search finds it → vector search finds it → hybrid search finds it
