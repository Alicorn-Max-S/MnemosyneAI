"""Intake: fast-path message write and derive-task enqueue."""

import logging

from mnemosyne.db.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


async def ingest_message(
    session_id: str,
    peer_id: str,
    role: str,
    content: str,
    db: SQLiteStore,
) -> str:
    """Write a message and enqueue a derive task for user messages.

    Returns the message ID.
    """
    message = await db.add_message(session_id, peer_id, role, content)
    logger.info("Ingested message %s (role=%s)", message.id, role)

    if role == "user":
        preceding = await db.get_recent_context(session_id, n_turns=3)
        # Filter out the just-written message — we want preceding turns only
        preceding_turns = [
            {"id": m.id, "role": m.role, "content": m.content}
            for m in preceding
            if m.id != message.id
        ]
        await db.enqueue_task(
            "derive",
            payload={
                "message_id": message.id,
                "session_id": session_id,
                "peer_id": peer_id,
                "content": content,
                "preceding_turns": preceding_turns,
            },
        )
        logger.info("Enqueued derive task for message %s", message.id)

    return message.id
