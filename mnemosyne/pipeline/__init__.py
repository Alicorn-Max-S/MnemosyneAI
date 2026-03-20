"""Pipeline: Worker queue poller, Intake fast-path ingestion, and Deriver LLM extraction."""

from mnemosyne.db.sqlite_store import SQLiteStore
from mnemosyne.pipeline.deriver import Deriver
from mnemosyne.pipeline.handlers import handle_derive
from mnemosyne.pipeline.intake import ingest_message
from mnemosyne.pipeline.worker import Worker
from mnemosyne.vectors.embedder import Embedder
from mnemosyne.vectors.zvec_store import ZvecStore


def create_worker(
    db: SQLiteStore,
    deriver: Deriver,
    embedder: Embedder,
    zvec: ZvecStore,
    linker=None,
) -> Worker:
    """Create a Worker with the derive handler wired up."""
    async def _handle(task):
        await handle_derive(task, db, deriver, embedder, zvec, linker=linker)
    return Worker(db, {"derive": _handle})


__all__ = ["Deriver", "Worker", "create_worker", "handle_derive", "ingest_message"]
