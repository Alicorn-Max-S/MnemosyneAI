"""Handlers: task-type callbacks wired into the Worker."""

import asyncio
import logging
import time

from mnemosyne.db.sqlite_store import SQLiteStore
from mnemosyne.models import TaskItem
from mnemosyne.pipeline.deriver import Deriver
from mnemosyne.vectors.embedder import Embedder
from mnemosyne.vectors.zvec_store import ZvecStore

logger = logging.getLogger(__name__)


async def handle_derive(
    task: TaskItem,
    db: SQLiteStore,
    deriver: Deriver,
    embedder: Embedder,
    zvec: ZvecStore,
) -> None:
    """Extract, score, embed, and persist notes from a user message.

    Reads message data from task.payload, runs the Deriver extract/score
    pipeline, embeds each scored note, and writes to SQLite + Zvec.
    """
    payload = task.payload
    message_id = payload["message_id"]
    session_id = payload["session_id"]
    peer_id = payload["peer_id"]
    content = payload["content"]
    preceding_turns = payload["preceding_turns"]

    t0 = time.monotonic()

    # Extract atomic facts
    notes = await deriver.extract(content, preceding_turns)
    t_extract = time.monotonic()
    if not notes:
        logger.info(
            "Task %s: extraction returned 0 notes (%.3fs)",
            task.id, t_extract - t0,
        )
        return

    # Score extracted notes
    scored_notes = await deriver.score(notes)
    t_score = time.monotonic()
    if not scored_notes:
        logger.info(
            "Task %s: scoring returned 0 notes (extract=%.3fs, score=%.3fs)",
            task.id, t_extract - t0, t_score - t_extract,
        )
        return

    # Persist each scored note
    created = 0
    for note in scored_notes:
        embedding = await asyncio.to_thread(embedder.embed_document, note["text"])

        note_obj = await db.create_note(
            peer_id,
            content=note["text"],
            session_id=session_id,
            source_message_id=message_id,
            note_type="observation",
            provenance=note.get("provenance", "organic"),
            durability=note.get("durability", "contextual"),
            emotional_weight=note.get("emotional_weight", 0.5),
            keywords=note.get("keywords", []),
            tags=note.get("tags", []),
            context_description=note.get("context_description"),
            is_buffered=True,
        )

        try:
            await asyncio.to_thread(zvec.insert, note_obj.id, embedding)
            await db.update_note(note_obj.id, zvec_id=note_obj.id)
        except Exception:
            logger.warning(
                "Zvec insert failed for note %s, skipping vector index",
                note_obj.id,
            )

        created += 1

    t_persist = time.monotonic()
    logger.info(
        "Task %s: created %d notes (extract=%.3fs, score=%.3fs, persist=%.3fs)",
        task.id, created,
        t_extract - t0, t_score - t_extract, t_persist - t_score,
    )
