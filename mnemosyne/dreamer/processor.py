"""Processes Gemini batch results back into SQLite and Zvec stores."""

import asyncio
import json
import logging
import sqlite3

from mnemosyne.config import (
    DEFAULT_CONFIDENCE_INFERENCE,
    DURABILITY_CONTEXTUAL,
    LINK_TYPES,
    NOTE_TYPE_INFERENCE,
    PROVENANCE_INFERRED,
)
from mnemosyne.db.sqlite_store import SQLiteStore
from mnemosyne.vectors.embedder import Embedder
from mnemosyne.vectors.zvec_store import ZvecStore

logger = logging.getLogger(__name__)


class DreamerProcessor:
    """Processes Gemini Batch API results into persistent stores.

    All methods are best-effort: individual failures are logged but never
    raised. Return values reflect partial success counts.
    """

    def __init__(
        self, db: SQLiteStore, embedder: Embedder, zvec: ZvecStore
    ) -> None:
        """Initialize with database, embedder, and vector store references."""
        self._db = db
        self._embedder = embedder
        self._zvec = zvec

    async def process_links(self, results: list[dict]) -> int:
        """Parse link generation results and create links in SQLite.

        Args:
            results: Parsed JSON dicts from Gemini, each containing a
                ``links`` list.

        Returns:
            Number of links successfully created.
        """
        count = 0
        for result in results:
            links = result.get("links", [])
            for link_data in links:
                try:
                    source_id = link_data.get("source_id")
                    target_id = link_data.get("target_id")
                    link_type = link_data.get("link_type")
                    strength = link_data.get("strength", 0.5)

                    if not source_id or not target_id or not link_type:
                        logger.debug("Skipping link with missing fields: %s", link_data)
                        continue

                    if link_type not in LINK_TYPES:
                        logger.debug("Skipping invalid link type: %s", link_type)
                        continue

                    strength = max(0.0, min(1.0, float(strength)))

                    await self._db.create_link(
                        source_note_id=source_id,
                        target_note_id=target_id,
                        link_type=link_type,
                        strength=strength,
                    )
                    count += 1
                except sqlite3.IntegrityError:
                    logger.debug(
                        "Duplicate link %s→%s (%s), skipping",
                        link_data.get("source_id"),
                        link_data.get("target_id"),
                        link_data.get("link_type"),
                    )
                except Exception:
                    logger.warning(
                        "Failed to create link: %s", link_data, exc_info=True
                    )

        logger.info("Processed links: %d created", count)
        return count

    async def process_patterns(self, results: list[dict], peer_id: str) -> int:
        """Create inference notes from detected patterns.

        Each pattern becomes a Note with ``note_type="inference"``,
        ``provenance="inferred"``, ``durability="contextual"``.

        Args:
            results: Parsed JSON dicts from Gemini, each containing a
                ``patterns`` list.
            peer_id: The peer these patterns belong to.

        Returns:
            Number of inference notes successfully created.
        """
        count = 0
        for result in results:
            patterns = result.get("patterns", [])
            for pattern in patterns:
                try:
                    content = pattern.get("content")
                    if not content:
                        continue

                    keywords = pattern.get("keywords", [])
                    supporting_ids = pattern.get("supporting_note_ids", [])

                    # Create inference note
                    note = await self._db.create_note(
                        peer_id=peer_id,
                        content=content,
                        keywords=keywords,
                        note_type=NOTE_TYPE_INFERENCE,
                        provenance=PROVENANCE_INFERRED,
                        durability=DURABILITY_CONTEXTUAL,
                        confidence=DEFAULT_CONFIDENCE_INFERENCE,
                        is_buffered=False,
                    )

                    # Embed and store in Zvec
                    embedding = await asyncio.to_thread(
                        self._embedder.embed_document, content
                    )
                    await asyncio.to_thread(
                        self._zvec.insert, note.id, embedding
                    )

                    # Create derived_from links to supporting notes
                    for supporting_id in supporting_ids:
                        try:
                            await self._db.create_link(
                                source_note_id=note.id,
                                target_note_id=supporting_id,
                                link_type="derived_from",
                            )
                        except Exception:
                            logger.debug(
                                "Failed to create derived_from link to %s",
                                supporting_id,
                            )

                    count += 1
                except Exception:
                    logger.warning(
                        "Failed to process pattern: %s",
                        pattern.get("content", "<unknown>")[:80],
                        exc_info=True,
                    )

        logger.info("Processed patterns: %d created", count)
        return count

    async def process_contradictions(self, results: list[dict]) -> int:
        """Create 'contradicts' links between conflicting notes.

        Args:
            results: Parsed JSON dicts from Gemini, each containing a
                ``contradictions`` list.

        Returns:
            Number of contradiction links successfully created.
        """
        count = 0
        for result in results:
            contradictions = result.get("contradictions", [])
            for contradiction in contradictions:
                try:
                    note_id_a = contradiction.get("note_id_a")
                    note_id_b = contradiction.get("note_id_b")
                    description = contradiction.get("description", "")

                    if not note_id_a or not note_id_b:
                        continue

                    await self._db.create_link(
                        source_note_id=note_id_a,
                        target_note_id=note_id_b,
                        link_type="contradicts",
                        strength=1.0,
                        metadata={"description": description},
                    )
                    count += 1
                except sqlite3.IntegrityError:
                    logger.debug(
                        "Duplicate contradiction link %s→%s, skipping",
                        contradiction.get("note_id_a"),
                        contradiction.get("note_id_b"),
                    )
                except Exception:
                    logger.warning(
                        "Failed to create contradiction link: %s",
                        contradiction,
                        exc_info=True,
                    )

        logger.info("Processed contradictions: %d created", count)
        return count

    async def process_profile(self, result: dict, peer_id: str) -> None:
        """Update the peer's static profile from Gemini results.

        Args:
            result: Parsed JSON dict containing profile sections.
            peer_id: The peer to update.
        """
        try:
            # Handle both {"profile": {...}} and direct section dict
            profile = result.get("profile", result)

            if not isinstance(profile, dict):
                logger.warning("Invalid profile format: expected dict, got %s", type(profile))
                return

            await self._db.update_peer(peer_id, static_profile=profile)
            logger.info("Updated static profile for peer %s", peer_id)
        except Exception:
            logger.warning(
                "Failed to update profile for peer %s", peer_id, exc_info=True
            )
