"""ColBERT reranker using pylate for pre-computed token embeddings and MaxSim scoring."""

import asyncio
import logging

import numpy as np

from mnemosyne.config import COLBERT_MODEL, COLBERT_TOKEN_DIM, COLBERT_TOP_N

logger = logging.getLogger(__name__)


class ColBERTReranker:
    """Reranks retrieval candidates using pre-computed ColBERT token embeddings.

    Write-time methods (encode_document, encode_documents) are sync — callers
    wrap them in asyncio.to_thread().  The read-time method (rerank) is async
    because it fetches pre-computed tokens from the database before scoring.
    """

    def __init__(self, model_name: str = COLBERT_MODEL) -> None:
        """Store model name for lazy loading.

        Args:
            model_name: HuggingFace model identifier for the ColBERT model.
        """
        self._model_name = model_name
        self._model = None

    def _ensure_loaded(self) -> bool:
        """Load the model on first use. Returns True if model is available."""
        if self._model is not None:
            return True
        try:
            from pylate.models import ColBERT

            self._model = ColBERT(model_name_or_path=self._model_name)
            logger.info("Loaded ColBERT model: %s", self._model_name)
            return True
        except Exception:
            logger.warning("Failed to load ColBERT model, functionality disabled")
            return False

    def is_loaded(self) -> bool:
        """Return whether the model is loaded. Does NOT trigger lazy load."""
        return self._model is not None

    # ── Write-time methods (sync) ──────────────────────────────────

    def encode_document(self, text: str) -> bytes:
        """Encode a document into ColBERT token-level embeddings.

        Args:
            text: The document text to encode.

        Returns:
            Serialized numpy array of shape (num_tokens, COLBERT_TOKEN_DIM) as bytes.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        if not self._ensure_loaded():
            raise RuntimeError("ColBERT model not available")

        result = self._model.encode(
            [text], is_query=False, convert_to_numpy=True,
        )
        return result[0].astype(np.float32).tobytes()

    def encode_documents(self, texts: list[str]) -> list[bytes]:
        """Batch encode documents into ColBERT token-level embeddings.

        Args:
            texts: List of document texts to encode.

        Returns:
            List of serialized numpy arrays as bytes, one per document.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        if not self._ensure_loaded():
            raise RuntimeError("ColBERT model not available")

        results = self._model.encode(
            texts, is_query=False, convert_to_numpy=True,
        )
        return [arr.astype(np.float32).tobytes() for arr in results]

    # ── Read-time method (async) ───────────────────────────────────

    async def rerank(
        self,
        query: str,
        candidate_note_ids: list[str],
        db: "SQLiteStore",
        top_n: int = COLBERT_TOP_N,
    ) -> list[tuple[str, float]]:
        """Rerank candidates using pre-computed token embeddings from SQLite.

        Args:
            query: The search query string.
            candidate_note_ids: List of note IDs to rerank.
            db: SQLiteStore instance to fetch pre-computed token embeddings.
            top_n: Maximum number of results to return.

        Returns:
            List of (note_id, score) tuples sorted by score descending.
            Notes without pre-computed tokens are ranked at the end with score 0.0.
        """
        if not candidate_note_ids:
            return []

        if not self._ensure_loaded():
            return [(nid, 0.0) for nid in candidate_note_ids]

        try:
            # Fetch pre-computed document tokens from SQLite
            tokens_map = await db.get_colbert_tokens(candidate_note_ids)

            # Split into notes with/without tokens
            ids_with_tokens = [nid for nid in candidate_note_ids if nid in tokens_map]
            ids_without_tokens = [nid for nid in candidate_note_ids if nid not in tokens_map]

            if not ids_with_tokens:
                return [(nid, 0.0) for nid in candidate_note_ids]

            # Reconstruct document token embeddings
            doc_embeddings = []
            for nid in ids_with_tokens:
                blob = tokens_map[nid]
                arr = np.frombuffer(blob, dtype=np.float32).copy().reshape(
                    -1, COLBERT_TOKEN_DIM
                )
                doc_embeddings.append(arr)

            # Encode query tokens (sync → offload to thread)
            def _encode_query():
                return self._model.encode(
                    [query], is_query=True, convert_to_numpy=True,
                )

            query_emb = await asyncio.to_thread(_encode_query)

            # Score via pylate.rank.rerank (sync → offload to thread)
            scored = await asyncio.to_thread(
                self._maxsim_rerank,
                ids_with_tokens,
                query_emb,
                doc_embeddings,
            )

            # Append notes without tokens at the end
            for nid in ids_without_tokens:
                scored.append((nid, 0.0))

            return scored[:top_n]

        except Exception:
            logger.warning("ColBERT reranking failed, returning original order")
            return [(nid, 0.0) for nid in candidate_note_ids]

    @staticmethod
    def _maxsim_rerank(
        doc_ids: list[str],
        query_embeddings: list,
        doc_embeddings: list,
    ) -> list[tuple[str, float]]:
        """Compute MaxSim scores and return ranked (id, score) tuples."""
        from pylate import rank

        result = rank.rerank(
            documents_ids=[doc_ids],
            queries_embeddings=query_embeddings,
            documents_embeddings=[doc_embeddings],
        )

        ranked = [(r["id"], float(r["score"])) for r in result[0]]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
