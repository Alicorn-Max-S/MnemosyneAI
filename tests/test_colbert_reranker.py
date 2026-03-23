"""Tests for ColBERT reranker with pylate and pre-computed token embeddings."""

import numpy as np
import pytest

from mnemosyne.config import COLBERT_TOKEN_DIM
from mnemosyne.intelligence.reranker import ColBERTReranker


@pytest.fixture(scope="module")
def reranker():
    """Module-scoped reranker (expensive model load)."""
    r = ColBERTReranker()
    r._ensure_loaded()
    return r


class TestEncodeDocument:
    def test_returns_bytes(self, reranker):
        """encode_document returns bytes."""
        blob = reranker.encode_document("The quick brown fox")
        assert isinstance(blob, bytes)
        assert len(blob) > 0

    def test_correct_shape(self, reranker):
        """Encoded blob reconstructs to (num_tokens, COLBERT_TOKEN_DIM) array."""
        blob = reranker.encode_document("Hello world, this is a test")
        arr = np.frombuffer(blob, dtype=np.float32)
        assert arr.size % COLBERT_TOKEN_DIM == 0
        num_tokens = arr.size // COLBERT_TOKEN_DIM
        assert num_tokens > 0
        reshaped = arr.reshape(num_tokens, COLBERT_TOKEN_DIM)
        assert reshaped.shape[1] == COLBERT_TOKEN_DIM

    def test_different_texts_different_tokens(self, reranker):
        """Different texts produce different token embeddings."""
        blob1 = reranker.encode_document("cats are fluffy")
        blob2 = reranker.encode_document("quantum physics experiments")
        assert blob1 != blob2


class TestEncodeDocuments:
    def test_batch_returns_correct_count(self, reranker):
        """Batch encoding returns one blob per input text."""
        texts = ["text one", "text two", "text three"]
        blobs = reranker.encode_documents(texts)
        assert len(blobs) == 3
        assert all(isinstance(b, bytes) for b in blobs)

    def test_batch_shapes_valid(self, reranker):
        """All blobs in batch have valid shapes."""
        blobs = reranker.encode_documents(["short", "a longer sentence here"])
        for blob in blobs:
            arr = np.frombuffer(blob, dtype=np.float32)
            assert arr.size % COLBERT_TOKEN_DIM == 0
            assert arr.size // COLBERT_TOKEN_DIM > 0


class TestStoreGetRoundTrip:
    async def test_round_trip_through_sqlite(self, reranker, store):
        """Encode → store_colbert_tokens → get_colbert_tokens preserves data."""
        peer = await store.create_peer("ColBERTRoundTrip")
        note = await store.create_note(peer.id, "Test content for ColBERT tokens")

        blob = reranker.encode_document(note.content)
        num_tokens = len(blob) // (4 * COLBERT_TOKEN_DIM)
        await store.store_colbert_tokens(note.id, blob, num_tokens)

        tokens = await store.get_colbert_tokens([note.id])
        assert note.id in tokens
        reconstructed = np.frombuffer(tokens[note.id], dtype=np.float32)
        original = np.frombuffer(blob, dtype=np.float32)
        assert np.allclose(reconstructed, original)

    async def test_missing_ids_omitted(self, store):
        """get_colbert_tokens omits IDs that don't have stored tokens."""
        tokens = await store.get_colbert_tokens(["nonexistent_id"])
        assert tokens == {}


class TestRerank:
    async def test_rerank_with_precomputed_tokens(self, reranker, store):
        """Rerank with pre-computed tokens returns scored note IDs."""
        peer = await store.create_peer("RerankPeer")
        notes = []
        for text in ["Cats love fish and tuna", "Dogs play fetch outdoors", "Birds fly south in winter"]:
            note = await store.create_note(peer.id, text)
            blob = reranker.encode_document(text)
            num_tokens = len(blob) // (4 * COLBERT_TOKEN_DIM)
            await store.store_colbert_tokens(note.id, blob, num_tokens)
            notes.append(note)

        ids = [n.id for n in notes]
        result = await reranker.rerank("cats and fish", ids, store)
        assert len(result) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in result)
        # All input IDs should appear in output
        result_ids = {r[0] for r in result}
        assert result_ids == set(ids)

    async def test_notes_without_tokens_ranked_at_end(self, reranker, store):
        """Notes without pre-computed tokens are appended at the end with score 0.0."""
        peer = await store.create_peer("NoTokenPeer")
        n1 = await store.create_note(peer.id, "Has tokens about dogs and pets")
        n2 = await store.create_note(peer.id, "No tokens about cats")

        # Only store tokens for n1
        blob = reranker.encode_document("Has tokens about dogs and pets")
        num_tokens = len(blob) // (4 * COLBERT_TOKEN_DIM)
        await store.store_colbert_tokens(n1.id, blob, num_tokens)

        result = await reranker.rerank("dogs", [n1.id, n2.id], store)
        ranked_ids = [r[0] for r in result]
        # n2 (no tokens) should be at the end
        assert ranked_ids[-1] == n2.id
        # n2 score should be 0.0
        n2_score = next(score for nid, score in result if nid == n2.id)
        assert n2_score == 0.0

    async def test_top_n_respected(self, reranker, store):
        """Results truncated to top_n."""
        peer = await store.create_peer("TopNPeer")
        ids = []
        for i in range(10):
            text = f"Document number {i} about various interesting topics"
            note = await store.create_note(peer.id, text)
            blob = reranker.encode_document(text)
            num_tokens = len(blob) // (4 * COLBERT_TOKEN_DIM)
            await store.store_colbert_tokens(note.id, blob, num_tokens)
            ids.append(note.id)

        result = await reranker.rerank("topics", ids, store, top_n=3)
        assert len(result) == 3

    async def test_empty_candidates(self, reranker, store):
        """Empty candidates list returns empty result."""
        result = await reranker.rerank("any query", [], store)
        assert result == []

    async def test_scores_are_floats(self, reranker, store):
        """All scores are float type."""
        peer = await store.create_peer("FloatScorePeer")
        notes = []
        for text in ["Apples are red fruit", "Bananas are yellow"]:
            note = await store.create_note(peer.id, text)
            blob = reranker.encode_document(text)
            num_tokens = len(blob) // (4 * COLBERT_TOKEN_DIM)
            await store.store_colbert_tokens(note.id, blob, num_tokens)
            notes.append(note)

        result = await reranker.rerank("fruit", [n.id for n in notes], store)
        assert all(isinstance(score, float) for _, score in result)


class TestGracefulFailure:
    async def test_model_load_failure_returns_fallback(self, store):
        """Bad model name → rerank returns all candidates with score 0.0."""
        r = ColBERTReranker(model_name="nonexistent/model-that-wont-load")
        peer = await store.create_peer("FailPeer")
        n = await store.create_note(peer.id, "test note content")

        result = await r.rerank("query", [n.id], store)
        assert len(result) == 1
        assert result[0][0] == n.id
        assert result[0][1] == 0.0

    def test_encode_document_raises_without_model(self):
        """encode_document raises RuntimeError when model can't load."""
        r = ColBERTReranker(model_name="nonexistent/model-that-wont-load")
        with pytest.raises(RuntimeError):
            r.encode_document("test text")
