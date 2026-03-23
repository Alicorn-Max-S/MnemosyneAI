"""Handlers: task-type callbacks wired into the Worker."""

import asyncio
import logging
import time

from mnemosyne.config import COLBERT_TOKEN_DIM
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
    linker=None,
    colbert_reranker=None,
    magma_graph=None,
) -> None:
    """Extract, score, embed, and persist notes from a user message.

    Reads message data from task.payload, runs the Deriver extract/score
    pipeline, embeds each scored note, and writes to SQLite + Zvec.
    If a linker is provided, generates semantic links for each new note.
    If colbert_reranker is provided, pre-computes token embeddings for ColBERT.
    If magma_graph is provided, extracts entities for the MAGMA graph.
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

    # Batch-embed all texts at once (faster per-text than individual calls)
    texts = [note["text"] for note in scored_notes]
    embeddings = await asyncio.to_thread(embedder.embed_documents, texts)

    # Create notes in SQLite and collect Zvec items
    zvec_items: list[tuple[str, list[float]]] = []
    note_objs: list[tuple] = []
    created = 0
    for note, embedding in zip(scored_notes, embeddings):
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
        zvec_items.append((note_obj.id, embedding))
        note_objs.append((note_obj, embedding))
        created += 1

    # Batch insert to Zvec (single HNSW optimize instead of per-note)
    if zvec_items:
        try:
            await asyncio.to_thread(zvec.insert_batch, zvec_items)
            for note_id, _ in zvec_items:
                await db.update_note(note_id, zvec_id=note_id)
        except Exception:
            logger.warning(
                "Zvec batch insert failed for %d notes, skipping vector index",
                len(zvec_items),
            )

    # Generate semantic links for each new note
    if linker is not None:
        for note_obj, embedding in note_objs:
            try:
                await linker.generate_links(note_obj, embedding)
            except Exception:
                logger.warning("Link generation failed for note %s", note_obj.id)

    # Pre-compute ColBERT token embeddings
    if colbert_reranker is not None:
        try:
            texts = [note_obj.content for note_obj, _ in note_objs]
            token_blobs = await asyncio.to_thread(
                colbert_reranker.encode_documents, texts
            )
            for (note_obj, _), blob in zip(note_objs, token_blobs):
                num_tokens = len(blob) // (4 * COLBERT_TOKEN_DIM)
                await db.store_colbert_tokens(note_obj.id, blob, num_tokens)
        except Exception:
            logger.warning(
                "ColBERT token pre-computation failed for %d notes",
                len(note_objs),
            )

    # Extract entities for MAGMA graph
    if magma_graph is not None:
        for note_obj, _ in note_objs:
            try:
                entities = magma_graph.extract_entities(note_obj.content)
                if entities:
                    await magma_graph.add_note_entities(note_obj, entities)
            except Exception:
                logger.warning("Entity extraction failed for note %s", note_obj.id)

    t_persist = time.monotonic()
    logger.info(
        "Task %s: created %d notes (extract=%.3fs, score=%.3fs, persist=%.3fs)",
        task.id, created,
        t_extract - t0, t_score - t_extract, t_persist - t_score,
    )
