"""Coordinates a full Dreamer cycle: dedup → batch → process → graph."""

import asyncio
import logging
from dataclasses import dataclass, field

import numpy as np

from mnemosyne.db.sqlite_store import SQLiteStore
from mnemosyne.dreamer.dedup import DedupProcessor
from mnemosyne.dreamer.gemini_client import GeminiClient, GeminiError
from mnemosyne.dreamer.processor import DreamerProcessor
from mnemosyne.dreamer.task_builder import (
    build_contradiction_requests,
    build_link_requests,
    build_pattern_requests,
    build_profile_request,
)
from mnemosyne.graph.magma import MAGMAGraph
from mnemosyne.models import Note
from mnemosyne.vectors.embedder import Embedder
from mnemosyne.vectors.zvec_store import ZvecStore

logger = logging.getLogger(__name__)

_CONTRADICTION_COSINE_THRESHOLD = 0.80


@dataclass
class CycleResult:
    """Summary of a single Dreamer cycle."""

    notes_deduped: int = 0
    links_created: int = 0
    patterns_found: int = 0
    contradictions_found: int = 0
    profile_updated: bool = False


class DreamerOrchestrator:
    """Coordinates the full Dreamer background processing cycle."""

    def __init__(
        self,
        db: SQLiteStore,
        embedder: Embedder,
        zvec: ZvecStore,
        gemini_client: GeminiClient,
        deriver: object,
    ) -> None:
        """Initialize with all required dependencies.

        Args:
            db: SQLite store for all CRUD operations.
            embedder: Embedding model for vectorization.
            zvec: Vector store for similarity search.
            gemini_client: Gemini Batch API client.
            deriver: Deriver API reference (reserved for future use).
        """
        self._db = db
        self._embedder = embedder
        self._zvec = zvec
        self._gemini = gemini_client
        self._deriver = deriver
        self._dedup = DedupProcessor(db, embedder)
        self._processor = DreamerProcessor(db, embedder, zvec)
        self._graph = MAGMAGraph(db)

    async def run_cycle(self, peer_id: str) -> CycleResult:
        """Run one full Dreamer cycle for a peer.

        Steps: dedup → find contradictions → build batch → submit →
        poll → process results → MAGMA update.

        Returns a CycleResult with counts. On Gemini failure, returns
        partial result with dedup counts only.
        """
        result = CycleResult()

        # Step 1: Get buffered notes before dedup (need IDs to re-fetch after)
        pre_dedup_notes = await self._db.get_buffered_notes(peer_id)
        pre_dedup_ids = [n.id for n in pre_dedup_notes]

        # Step 2: Run dedup
        dedup_result = await self._dedup.run(peer_id)
        result.notes_deduped = dedup_result.notes_merged
        logger.info(
            "Dedup complete for peer %s: %d processed, %d merged",
            peer_id,
            dedup_result.notes_processed,
            dedup_result.notes_merged,
        )

        # Step 3: Re-fetch notes to find canonical survivors
        if not pre_dedup_ids:
            logger.info("No buffered notes for peer %s, returning early", peer_id)
            return result

        refetched = await self._db.get_notes_by_ids(pre_dedup_ids)
        canonical_notes = [
            n for n in refetched.values() if n.canonical_note_id is None
        ]

        if not canonical_notes:
            logger.info("No canonical notes after dedup for peer %s", peer_id)
            return result

        # Steps 4-6: Gemini batch (wrapped for graceful failure)
        try:
            await self._run_batch(peer_id, canonical_notes, result)
        except GeminiError:
            logger.warning(
                "Gemini batch failed for peer %s, returning partial result",
                peer_id,
                exc_info=True,
            )
        except Exception:
            logger.warning(
                "Unexpected error in batch processing for peer %s",
                peer_id,
                exc_info=True,
            )

        # Step 7: MAGMA entity graph update
        try:
            await self._graph.load(peer_id)
            for note in canonical_notes:
                entities = self._graph.extract_entities(note.content)
                if entities:
                    await self._graph.add_note_entities(note, entities)
            logger.info("MAGMA graph updated for peer %s", peer_id)
        except Exception:
            logger.warning(
                "MAGMA graph update failed for peer %s", peer_id, exc_info=True
            )

        return result

    async def _run_batch(
        self, peer_id: str, notes: list[Note], result: CycleResult
    ) -> None:
        """Build, submit, poll, and process a Gemini batch job."""
        # Find contradiction candidates (cosine > 0.80)
        candidate_pairs = await self._find_contradiction_candidates(notes)

        # Gather existing links for link generation
        existing_links = []
        for note in notes:
            existing_links.extend(await self._db.get_links(note.id))

        # Build all batch requests
        link_requests = build_link_requests(notes, existing_links)
        sessions = await self._db.list_sessions(peer_id)
        pattern_requests = build_pattern_requests(notes, sessions)
        contradiction_requests = build_contradiction_requests(candidate_pairs)
        permanent_notes = await self._db.get_permanent_notes(peer_id)
        peer = await self._db.get_peer(peer_id)
        current_profile = peer.static_profile if peer else None
        profile_request = build_profile_request(permanent_notes, current_profile)

        # Combine into single batch with index tracking
        all_requests = (
            link_requests
            + pattern_requests
            + contradiction_requests
            + [profile_request]
        )

        n_link = len(link_requests)
        n_pattern = len(pattern_requests)
        n_contradiction = len(contradiction_requests)

        # Submit and poll
        job_name = await self._gemini.submit_batch(
            all_requests, f"dreamer-{peer_id}"
        )
        job = await self._gemini.poll_until_done(job_name)
        all_results = await self._gemini.get_results(job)

        # Route results to processors by index boundaries
        link_results = all_results[:n_link]
        pattern_results = all_results[n_link : n_link + n_pattern]
        contradiction_results = all_results[
            n_link + n_pattern : n_link + n_pattern + n_contradiction
        ]
        profile_idx = n_link + n_pattern + n_contradiction
        profile_result = all_results[profile_idx] if profile_idx < len(all_results) else None

        # Process each result type (best-effort per type)
        try:
            result.links_created = await self._processor.process_links(link_results)
        except Exception:
            logger.warning("Link processing failed", exc_info=True)

        try:
            result.patterns_found = await self._processor.process_patterns(
                pattern_results, peer_id
            )
        except Exception:
            logger.warning("Pattern processing failed", exc_info=True)

        try:
            result.contradictions_found = await self._processor.process_contradictions(
                contradiction_results
            )
        except Exception:
            logger.warning("Contradiction processing failed", exc_info=True)

        if profile_result is not None:
            try:
                await self._processor.process_profile(profile_result, peer_id)
                result.profile_updated = True
            except Exception:
                logger.warning("Profile processing failed", exc_info=True)

    async def _find_contradiction_candidates(
        self, notes: list[Note]
    ) -> list[tuple[Note, Note]]:
        """Find note pairs with cosine similarity > 0.80 for contradiction detection."""
        if len(notes) < 2:
            return []

        contents = [n.content for n in notes]
        embeddings = await asyncio.to_thread(
            self._embedder.embed_documents, contents
        )
        emb_matrix = np.array(embeddings)

        # Cosine similarity (embeddings are L2-normalized)
        similarity = emb_matrix @ emb_matrix.T

        pairs: list[tuple[Note, Note]] = []
        for i in range(len(notes)):
            for j in range(i + 1, len(notes)):
                if similarity[i, j] > _CONTRADICTION_COSINE_THRESHOLD:
                    pairs.append((notes[i], notes[j]))

        logger.info("Found %d contradiction candidate pairs", len(pairs))
        return pairs
