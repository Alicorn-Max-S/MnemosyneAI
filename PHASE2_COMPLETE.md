# Phase 2 Complete: Write Pipeline

## New Files

| File | Purpose |
|------|---------|
| `mnemosyne/pipeline/handlers.py` | `handle_derive` — extracts, scores, embeds, and persists notes from user messages |
| `tests/test_write_pipeline.py` | Integration tests covering full round-trip, retry, graceful Zvec failure, empty extraction, and assistant-no-derive |

## Modified Files

| File | Change |
|------|--------|
| `mnemosyne/pipeline/__init__.py` | Added `create_worker` factory and `handle_derive` export |

## Worker Startup Example

```python
import asyncio
from mnemosyne.config import WORKER_POLL_INTERVAL
from mnemosyne.db.sqlite_store import SQLiteStore
from mnemosyne.pipeline import Deriver, create_worker
from mnemosyne.vectors.embedder import Embedder
from mnemosyne.vectors.zvec_store import ZvecStore

async def main():
    db = SQLiteStore("./data/mnemosyne.db")
    await db.initialize()

    embedder = Embedder()
    zvec = ZvecStore("./data")
    deriver = Deriver(api_key=os.environ["NOUSRESEARCH_API_KEY"])

    worker = create_worker(db, deriver, embedder, zvec)
    await worker.run(poll_interval=WORKER_POLL_INTERVAL)

asyncio.run(main())
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NOUSRESEARCH_API_KEY` | Yes | API key for the NousResearch LLM endpoint |

## Verification

```bash
# Full test suite
pytest tests/ -v

# Pipeline integration tests only
pytest tests/test_write_pipeline.py -v
```

## Known Limitations

- Zvec inserts are per-note (no batching) — acceptable for low-throughput derive workloads.
- `importance` and `confidence` use SQLite defaults (0.0 and 0.8) — the scorer does not produce these fields yet.
- The worker is single-threaded; concurrent derive tasks are processed sequentially.
- No dead-letter recovery UI — dead-lettered tasks require manual SQL intervention.
