"""8-step retrieval pipeline for Mnemosyne with link expansion and ColBERT reranking."""

import asyncio
import logging
import re
from datetime import datetime, timezone

from mnemosyne.config import (
    COLBERT_RERANK_CANDIDATES,
    COLBERT_TOP_N,
    LINK_EXPANSION_DEPTH,
    LINK_EXPANSION_MAX,
    LINK_EXPANSION_TOP_SEEDS,
    RETRIEVAL_FTS_LIMIT,
    RETRIEVAL_VECTOR_LIMIT,
)
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

# Max FTS-only notes to embed on-the-fly for MMR dedup
_MMR_EMBED_CAP = 10


class Retriever:
    """Coordinates parallel search, fusion, scoring, MMR dedup, and access recording."""

    def __init__(
        self,
        sqlite_store: SQLiteStore,
        zvec_store: ZvecStore,
        embedder: Embedder,
        colbert_reranker=None,
    ) -> None:
        """Initialize with store and embedder references."""
        self._db = sqlite_store
        self._zvec = zvec_store
        self._embedder = embedder
        self._reranker = colbert_reranker

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Strip characters that are special in FTS5 query syntax."""
        cleaned = re.sub(r"[^\w\s]", " ", query)
        tokens = cleaned.split()
        return " ".join(tokens) if tokens else ""

    async def _mmr_dedup_results(
        self,
        results: list[RetrievalResult],
        vec_set: set[str],
    ) -> list[RetrievalResult]:
        """Apply MMR dedup using embeddings, embedding FTS-only notes on the fly."""
        sorted_ids = [r.note.id for r in results]

        # Collect embeddings: vector-sourced notes already have them in Zvec,
        # but we need to re-embed for MMR.  Batch-embed all candidate texts,
        # capping FTS-only notes to _MMR_EMBED_CAP to limit latency.
        fts_only_ids = [nid for nid in sorted_ids if nid not in vec_set]
        ids_to_embed = list(vec_set & set(sorted_ids)) + fts_only_ids[:_MMR_EMBED_CAP]

        texts = []
        id_order = []
        result_map = {r.note.id: r for r in results}
        for nid in ids_to_embed:
            r = result_map.get(nid)
            if r is not None:
                texts.append(r.note.content)
                id_order.append(nid)

        embeddings: dict[str, list[float]] = {}
        if texts:
            vecs = await asyncio.to_thread(self._embedder.embed_documents, texts)
            for nid, vec in zip(id_order, vecs):
                embeddings[nid] = vec

        deduped_ids = mmr_dedup(sorted_ids, embeddings)
        deduped_set = set(deduped_ids)
        return [r for r in results if r.note.id in deduped_set]

    async def retrieve(
        self,
        query: str,
        peer_id: str,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        """Execute the full retrieval pipeline with link expansion and optional ColBERT reranking."""
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
                return await self._db.fts_search_ranked(
                    sanitized, peer_id, limit=RETRIEVAL_FTS_LIMIT
                )
            except Exception:
                logger.warning("FTS search failed, degrading to vector-only")
                return []

        async def _safe_vec_search() -> list[str]:
            try:
                results = await asyncio.to_thread(
                    self._zvec.search, query_embedding, RETRIEVAL_VECTOR_LIMIT
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

        # 5. Hydrate — batch fetch and filter by peer_id
        notes_map = await self._db.get_notes_by_ids(all_ids)
        notes_map = {
            nid: note for nid, note in notes_map.items() if note.peer_id == peer_id
        }

        if not notes_map:
            return []

        # 6. Preliminary RRF for seed selection
        preliminary_rrf = rrf_fuse([fts_ids, vec_ids])
        seed_candidates = sorted(
            notes_map.keys(), key=lambda nid: preliminary_rrf.get(nid, 0.0), reverse=True
        )
        seed_ids = seed_candidates[:LINK_EXPANSION_TOP_SEEDS]

        # 7. Link expansion via get_linked_notes()
        link_ids: list[str] = []
        try:
            linked_notes = await self._db.get_linked_notes(
                seed_ids, depth=LINK_EXPANSION_DEPTH, max_per_seed=LINK_EXPANSION_MAX
            )
            for note in linked_notes:
                if note.peer_id != peer_id:
                    continue
                if note.id in notes_map:
                    continue
                notes_map[note.id] = note
                base = source_map.get(note.id, "")
                source_map[note.id] = f"{base}+link" if base else "link"
                link_ids.append(note.id)
        except Exception:
            logger.warning("Link expansion failed, using 2-list RRF")
            link_ids = []

        # 8. Full RRF fusion with 2 or 3 lists
        rrf_lists = [fts_ids, vec_ids]
        if link_ids:
            rrf_lists.append(link_ids)
        rrf_scores = rrf_fuse(rrf_lists)

        # Get total memories once for decay formula
        total_memories = await self._db.count_notes(peer_id)

        # 9. Composite score
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
                    composite_score=composite,
                    rrf_score=rrf,
                    decay_strength=decay,
                    provenance_weight=prov,
                    fatigue_factor=fatigue,
                    inference_discount=inf_disc,
                    colbert_score=None,
                    source=source_map.get(nid, "fts"),
                )
            )

        # 10. Sort by composite score descending
        results.sort(key=lambda r: r.score, reverse=True)

        # 11. ColBERT reranking OR MMR dedup
        if self._reranker is not None:
            try:
                candidates_for_rerank = results[:COLBERT_RERANK_CANDIDATES]
                candidate_ids = [r.note.id for r in candidates_for_rerank]
                reranked = await self._reranker.rerank(
                    query, candidate_ids, self._db, COLBERT_TOP_N
                )
                # Build lookup from rerank results
                rerank_map = {nid: score for nid, score in reranked}
                reranked_ids = [nid for nid, _ in reranked]

                # If no candidate had pre-computed tokens (all scores 0.0),
                # fall back to MMR dedup instead of using useless scores.
                if not any(s > 0.0 for s in rerank_map.values()):
                    logger.debug("No pre-computed ColBERT tokens, falling back to MMR")
                    results = await self._mmr_dedup_results(results, vec_set)
                else:
                    result_map = {r.note.id: r for r in candidates_for_rerank}

                    reranked_results: list[RetrievalResult] = []
                    for nid in reranked_ids:
                        r = result_map.get(nid)
                        if r is not None:
                            reranked_results.append(
                                RetrievalResult(
                                    note=r.note,
                                    score=rerank_map[nid],
                                    composite_score=r.composite_score,
                                    rrf_score=r.rrf_score,
                                    decay_strength=r.decay_strength,
                                    provenance_weight=r.provenance_weight,
                                    fatigue_factor=r.fatigue_factor,
                                    inference_discount=r.inference_discount,
                                    colbert_score=rerank_map[nid],
                                    source=r.source,
                                )
                            )
                    results = reranked_results
            except Exception:
                logger.warning("ColBERT reranking failed, falling back to MMR dedup")
                results = await self._mmr_dedup_results(results, vec_set)
        else:
            # Phase 3 fallback — MMR dedup
            results = await self._mmr_dedup_results(results, vec_set)

        # 12. Truncate
        results = results[:limit]

        # 13. Record access
        returned_ids = [r.note.id for r in results]
        if returned_ids:
            await self._db.record_access(returned_ids)

        return results
