"""Tests for the Embedder wrapper around nomic-embed-text-v1.5."""

import pytest

from mnemosyne.vectors.embedder import Embedder


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    """Module-scoped embedder — model loading is expensive."""
    return Embedder()


def test_model_loads(embedder: Embedder) -> None:
    """Embedder() should load without raising."""
    assert embedder is not None


def test_embed_document_dim(embedder: Embedder) -> None:
    """embed_document should return a list of exactly 384 floats."""
    vec = embedder.embed_document("The quick brown fox jumps over the lazy dog.")
    assert isinstance(vec, list)
    assert len(vec) == 384
    assert all(isinstance(v, float) for v in vec)


def test_embed_query_dim(embedder: Embedder) -> None:
    """embed_query should return a list of exactly 384 floats."""
    vec = embedder.embed_query("quick brown fox")
    assert isinstance(vec, list)
    assert len(vec) == 384
    assert all(isinstance(v, float) for v in vec)


def test_embed_documents_batch(embedder: Embedder) -> None:
    """embed_documents should return one vector per input text."""
    texts = ["hello world", "goodbye world", "test embedding"]
    vecs = embedder.embed_documents(texts)
    assert len(vecs) == 3
    assert all(len(v) == 384 for v in vecs)


def test_different_texts_different_embeddings(embedder: Embedder) -> None:
    """Semantically different texts should produce different embeddings."""
    v1 = embedder.embed_document("The cat sat on the mat.")
    v2 = embedder.embed_document("Quantum chromodynamics describes the strong force.")
    assert v1 != v2


def test_same_text_identical_embeddings(embedder: Embedder) -> None:
    """The same text should produce identical embeddings."""
    text = "Reproducibility is important."
    v1 = embedder.embed_document(text)
    v2 = embedder.embed_document(text)
    assert v1 == v2


def test_doc_vs_query_prefix_differs(embedder: Embedder) -> None:
    """Document and query prefixes should produce different embeddings for the same text."""
    text = "cats"
    v_doc = embedder.embed_document(text)
    v_query = embedder.embed_query(text)
    assert v_doc != v_query


def test_backend_property(embedder: Embedder) -> None:
    """backend should return 'onnx' or 'torch'."""
    assert embedder.backend in ("onnx", "torch")


def test_onnx_preferred(embedder: Embedder) -> None:
    """If onnxruntime and optimum are importable, ONNX backend should be used."""
    try:
        import onnxruntime  # noqa: F401
        import optimum.onnxruntime  # noqa: F401
        assert embedder.backend == "onnx"
    except ImportError:
        pytest.skip("onnxruntime or optimum not installed")


def test_dimension_property(embedder: Embedder) -> None:
    """dimension should return 384."""
    assert embedder.dimension == 384
