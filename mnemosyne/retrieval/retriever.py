"""8-step retrieval pipeline for Mnemosyne."""

import asyncio
import logging
import re
from datetime import datetime, timezone

from mnemosyne.db.sqlite_store import SQLiteStore
from mnemosyne.models import RetrievalResult
from mnemosyne.retrieval.fusion import mmr_dedup, rrf_fuse
from mnemosyne.retrieval.scorer import (
    compute_composite_score,
    compute_decay_strength,
    compute_inference_discount,
    compute_provenance_weight,
    compute_surfacing_fatigue,
)
from mnemosyne.vectors.embedder import Embedder
from mnemosyne.vectors.zvec_store import ZvecStore

logger = logging.getLogger(__name__)


class Retriever:
    """Coordinates parallel search, fusion, scoring, MMR dedup, and access recording."""

    def __init__(
        self,
        sqlite_store: SQLiteStore,
        zvec_store: ZvecStore,
        embedder: Embedder,
    ) -> None:
        """Initialize with store and embedder references."""
        self._db = sqlite_store
        self._zvec = zvec_store
        self._embedder = embedder

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Strip characters that are special in FTS5 query syntax."""
        cleaned = re.sub(r"[^\w\s]", " ", query)
        tokens = cleaned.split()
        return " ".join(tokens) if tokens else ""

    async def retrieve(
        self,
        query: str,
        peer_id: str,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        """Execute the full 8-step retrieval pipeline."""
        # 1. Sanitize query
        sanitized = self._sanitize_fts_query(query)
        if not sanitized:
            return []

        # 2. Embed query
        query_embedding = await asyncio.to_thread(
            self._embedder.embed_query, query
        )

        # 3. Parallel search with graceful degradation
        async def _safe_fts_search() -> list[str]:
            try:
                return await self._db.fts_search_ranked(sanitized, peer_id, limit=30)
            except Exception:
                logger.warning("FTS search failed, degrading to vector-only")
                return []

        async def _safe_vec_search() -> list[str]:
            try:
                results = await asyncio.to_thread(
                    self._zvec.search, query_embedding, 30
                )
                return [r["id"] for r in results]
            except Exception:
                logger.warning("Vector search failed, degrading to FTS-only")
                return []

        fts_ids, vec_ids = await asyncio.gather(
            _safe_fts_search(), _safe_vec_search()
        )

        if not fts_ids and not vec_ids:
            return []

        # 4. Combine — build source map
        source_map: dict[str, str] = {}
        fts_set = set(fts_ids)
        vec_set = set(vec_ids)
        all_ids = list(dict.fromkeys(fts_ids + vec_ids))  # preserve order, dedup
        for nid in all_ids:
            in_fts = nid in fts_set
            in_vec = nid in vec_set
            if in_fts and in_vec:
                source_map[nid] = "both"
            elif in_fts:
                source_map[nid] = "fts"
            else:
                source_map[nid] = "vector"

        # 5. RRF fuse
        rrf_scores = rrf_fuse([fts_ids, vec_ids])

        # 6. Hydrate — batch fetch and filter by peer_id (Zvec has no peer filtering)
        notes_map = await self._db.get_notes_by_ids(all_ids)
        notes_map = {
            nid: note for nid, note in notes_map.items() if note.peer_id == peer_id
        }

        if not notes_map:
            return []

        # Get total memories once for decay formula
        total_memories = await self._db.count_notes(peer_id)

        # 7. Composite score
        now = datetime.now(timezone.utc)
        results: list[RetrievalResult] = []
        for nid, note in notes_map.items():
            rrf = rrf_scores.get(nid, 0.0)

            # Parse last_accessed_at or fall back to created_at
            ts_str = note.last_accessed_at or note.created_at
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            days_since_access = max((now - ts).total_seconds() / 86400.0, 0.0)

            decay = compute_decay_strength(
                note.importance, days_since_access, note.access_count, total_memories
            )
            prov = compute_provenance_weight(note.provenance)
            fatigue = compute_surfacing_fatigue(note.times_surfaced)
            inf_disc = compute_inference_discount(note.note_type)
            composite = compute_composite_score(rrf, decay, prov, fatigue, inf_disc)

            results.append(
                RetrievalResult(
                    note=note,
                    score=composite,
                    rrf_score=rrf,
                    decay_strength=decay,
                    provenance_weight=prov,
                    fatigue_factor=fatigue,
                    inference_discount=inf_disc,
                    source=source_map.get(nid, "fts"),
                )
            )

        # 8. Sort by composite score descending
        results.sort(key=lambda r: r.score, reverse=True)

        # 9. MMR dedup
        sorted_ids = [r.note.id for r in results]
        contents = [r.note.content for r in results]
        embeddings_list = await asyncio.to_thread(
            self._embedder.embed_documents, contents
        )
        embeddings_map = dict(zip(sorted_ids, embeddings_list))
        deduped_ids = mmr_dedup(sorted_ids, embeddings_map)
        deduped_set = set(deduped_ids)
        results = [r for r in results if r.note.id in deduped_set]

        # 10. Truncate
        results = results[:limit]

        # 11. Record access
        returned_ids = [r.note.id for r in results]
        if returned_ids:
            await self._db.record_access(returned_ids)

        return results
