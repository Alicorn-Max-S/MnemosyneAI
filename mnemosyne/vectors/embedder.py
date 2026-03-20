"""Embedder wrapping nomic-embed-text-v1.5 with ONNX-first, PyTorch fallback."""

import functools
import logging

from sentence_transformers import SentenceTransformer

from mnemosyne.config import EMBEDDING_DIM, EMBEDDING_DOC_PREFIX, EMBEDDING_MODEL, EMBEDDING_QUERY_PREFIX

logger = logging.getLogger(__name__)

# Keys that the nomic model config puts in model_args but that optimum's
# ORTModel._from_pretrained does not accept.  We strip them at the boundary.
_ONNX_UNSUPPORTED_KWARGS = frozenset({"safe_serialization"})


def _patch_onnx_loader() -> None:
    """Patch sentence_transformers' ONNX loader to strip kwargs that optimum rejects.

    The nomic model config includes ``safe_serialization`` in its ``model_args``.
    sentence-transformers passes these through to optimum's
    ``ORTModel._from_pretrained``, which does not accept them.  We intercept the
    call to strip unsupported keys.

    The patched function must be installed in every module that imported the
    original by name (``from … import load_onnx_model``), otherwise the local
    reference in that module still points to the unpatched version.
    """
    import importlib

    from sentence_transformers.backend import load as st_load

    original = st_load.load_onnx_model

    @functools.wraps(original)
    def _patched(model_name_or_path, config, task_name, **model_kwargs):
        for key in _ONNX_UNSUPPORTED_KWARGS:
            model_kwargs.pop(key, None)
        return original(model_name_or_path, config, task_name, **model_kwargs)

    # Patch the canonical definition
    st_load.load_onnx_model = _patched
    # Also patch the module-level reference imported by Transformer.py
    transformer_mod = importlib.import_module("sentence_transformers.models.Transformer")
    transformer_mod.load_onnx_model = _patched


class Embedder:
    """Thin wrapper around SentenceTransformer for nomic-embed-text-v1.5."""

    def __init__(self) -> None:
        """Load the embedding model, preferring ONNX backend."""
        try:
            _patch_onnx_loader()
            self._model = SentenceTransformer(
                EMBEDDING_MODEL,
                trust_remote_code=True,
                truncate_dim=EMBEDDING_DIM,
                backend="onnx",
                model_kwargs={"file_name": "model.onnx"},
            )
            self._backend = "onnx"
            logger.info("Loaded embedding model with ONNX backend")
        except Exception:
            logger.warning("ONNX backend unavailable, falling back to PyTorch")
            self._model = SentenceTransformer(
                EMBEDDING_MODEL,
                trust_remote_code=True,
                truncate_dim=EMBEDDING_DIM,
            )
            self._backend = "torch"
            logger.info("Loaded embedding model with PyTorch backend")

    @property
    def backend(self) -> str:
        """Return the active backend name ('onnx' or 'torch')."""
        return self._backend

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return EMBEDDING_DIM

    def embed_document(self, text: str) -> list[float]:
        """Embed a single document text with the document prefix."""
        prefixed = f"{EMBEDDING_DOC_PREFIX}{text}"
        vector = self._model.encode(prefixed, normalize_embeddings=True)
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple document texts with the document prefix."""
        prefixed = [f"{EMBEDDING_DOC_PREFIX}{t}" for t in texts]
        vectors = self._model.encode(prefixed, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a query text with the query prefix."""
        prefixed = f"{EMBEDDING_QUERY_PREFIX}{query}"
        vector = self._model.encode(prefixed, normalize_embeddings=True)
        return vector.tolist()
