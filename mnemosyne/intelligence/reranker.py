"""ColBERT reranker for semantic reranking of retrieval candidates."""

import logging

from mnemosyne.config import COLBERT_MODEL, COLBERT_TOP_N

logger = logging.getLogger(__name__)


class ColBERTReranker:
    """Reranks retrieval candidates using a ColBERT model via the rerankers library."""

    def __init__(self, model_name: str = COLBERT_MODEL) -> None:
        """Load the ColBERT reranker model.

        Args:
            model_name: HuggingFace model identifier for the ColBERT model.
        """
        self._ranker = None
        try:
            from rerankers import Reranker

            self._ranker = Reranker(model_name, model_type="colbert")
            logger.info("Loaded ColBERT reranker: %s", model_name)
        except Exception:
            logger.warning("Failed to load ColBERT reranker, reranking disabled")

    def is_loaded(self) -> bool:
        """Return whether the reranker model is loaded and ready."""
        return self._ranker is not None

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
        top_n: int = COLBERT_TOP_N,
    ) -> list[tuple[str, float]]:
        """Rerank candidates by semantic relevance to the query.

        Args:
            query: The search query string.
            candidates: List of (note_id, text) tuples to rerank.
            top_n: Maximum number of results to return.

        Returns:
            List of (note_id, score) tuples sorted by score descending.
        """
        if not candidates:
            return []

        if len(candidates) == 1:
            note_id, text = candidates[0]
            if self._ranker is not None:
                try:
                    results = self._ranker.rank(query=query, docs=[text])
                    score = float(results.results[0].score)
                    return [(note_id, score)]
                except Exception:
                    logger.warning("Reranker failed on single candidate")
                    return [(note_id, 1.0)]
            return [(note_id, 1.0)]

        if self._ranker is None:
            return [(cid, 0.0) for cid, _ in candidates]

        try:
            doc_texts = [text for _, text in candidates]
            results = self._ranker.rank(query=query, docs=doc_texts)

            ranked = []
            for result in results.results:
                idx = int(result.doc_id)
                note_id = candidates[idx][0]
                ranked.append((note_id, float(result.score)))

            ranked.sort(key=lambda x: x[1], reverse=True)
            return ranked[:top_n]
        except Exception:
            logger.warning("Reranker failed, returning original order")
            return [(cid, 0.0) for cid, _ in candidates]
