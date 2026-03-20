"""A-MEM linker for creating semantic links between similar notes."""

import asyncio
import logging

from mnemosyne.config import (
    LINK_DEFAULT_STRENGTH,
    LINK_MAX_CANDIDATES,
    LINK_SIMILARITY_THRESHOLD,
    LINK_STRENGTH_FROM_SIMILARITY,
)
from mnemosyne.db.sqlite_store import SQLiteStore
from mnemosyne.models import Link, Note
from mnemosyne.vectors.embedder import Embedder
from mnemosyne.vectors.zvec_store import ZvecStore

logger = logging.getLogger(__name__)


class Linker:
    """Creates and queries semantic links between notes based on embedding similarity."""

    def __init__(self, db: SQLiteStore, zvec: ZvecStore, embedder: Embedder) -> None:
        """Initialize with store references.

        Args:
            db: SQLite store for note and link persistence.
            zvec: Zvec vector store for similarity search.
            embedder: Embedding model wrapper.
        """
        self._db = db
        self._zvec = zvec
        self._embedder = embedder

    async def generate_links(self, note: Note, embedding: list[float]) -> list[Link]:
        """Generate semantic links between a note and its nearest neighbors.

        Uses Zvec scores directly as cosine similarity (embeddings are
        L2-normalized at the Embedder level). No re-embedding needed.

        Args:
            note: The source note to link from.
            embedding: The embedding vector for the source note.

        Returns:
            List of created Link objects. Returns [] on any error.
        """
        try:
            candidates = await asyncio.to_thread(
                self._zvec.search, embedding, LINK_MAX_CANDIDATES + 1
            )

            # Filter out self
            candidates = [c for c in candidates if c["id"] != note.id]

            if not candidates:
                return []

            candidate_ids = [c["id"] for c in candidates]
            notes_map = await self._db.get_notes_by_ids(candidate_ids)

            created_links: list[Link] = []
            for candidate in candidates:
                cand_id = candidate["id"]
                score = candidate["score"]

                if score < LINK_SIMILARITY_THRESHOLD:
                    continue

                cand_note = notes_map.get(cand_id)
                if cand_note is None or cand_note.peer_id != note.peer_id:
                    continue

                strength = score if LINK_STRENGTH_FROM_SIMILARITY else LINK_DEFAULT_STRENGTH

                try:
                    link = await self._db.create_link(
                        source_note_id=note.id,
                        target_note_id=cand_id,
                        link_type="semantic",
                        strength=strength,
                    )
                    created_links.append(link)
                except Exception:
                    logger.debug(
                        "Skipped duplicate link %s -> %s", note.id, cand_id
                    )

            return created_links
        except Exception:
            logger.warning("generate_links failed for note %s", note.id)
            return []

    async def find_neighbors(
        self, note_id: str, max_results: int = 5
    ) -> list[tuple[Note, float]]:
        """Find linked notes sorted by link strength.

        Args:
            note_id: The note ID to find neighbors for.
            max_results: Maximum number of neighbors to return.

        Returns:
            List of (Note, strength) tuples sorted by strength descending.
        """
        links = await self._db.get_links(note_id)
        if not links:
            return []

        # Collect linked IDs with strengths (handle both directions)
        linked: dict[str, float] = {}
        for link in links:
            if link.source_note_id == note_id:
                linked[link.target_note_id] = link.strength
            else:
                linked[link.source_note_id] = link.strength

        if not linked:
            return []

        notes_map = await self._db.get_notes_by_ids(list(linked.keys()))

        results = []
        for nid, strength in linked.items():
            n = notes_map.get(nid)
            if n is not None:
                results.append((n, strength))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
