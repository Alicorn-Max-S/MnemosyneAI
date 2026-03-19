"""Unit tests for Retriever with real SQLite, mocked Embedder and ZvecStore."""

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

from mnemosyne.models import RetrievalResult
from mnemosyne.retrieval.retriever import Retriever


# ── Helpers ──────────────────────────────────────────────────────


def _make_orthogonal_vectors(n: int, dim: int = 384) -> list[list[float]]:
    """Generate n distinct vectors that won't trigger MMR dedup."""
    rng = np.random.RandomState(42)
    vecs = []
    for i in range(n):
        v = rng.randn(dim).astype(np.float64)
        v = v / np.linalg.norm(v)
        vecs.append(v.tolist())
    return vecs


_VECTORS = _make_orthogonal_vectors(30)
_VECTOR_INDEX = 0


def _next_vector() -> list[float]:
    global _VECTOR_INDEX
    v = _VECTORS[_VECTOR_INDEX % len(_VECTORS)]
    _VECTOR_INDEX += 1
    return v


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_vector_index():
    global _VECTOR_INDEX
    _VECTOR_INDEX = 0


@pytest.fixture
def mock_embedder():
    """Embedder returning distinct 384-dim vectors per call."""
    emb = MagicMock()
    emb.embed_query.side_effect = lambda q: _next_vector()
    emb.embed_document.side_effect = lambda t: _next_vector()
    emb.embed_documents.side_effect = lambda texts: [_next_vector() for _ in texts]
    return emb


@pytest.fixture
def mock_zvec():
    """ZvecStore with controllable search results."""
    zv = MagicMock()
    zv.search.return_value = []
    return zv


@pytest.fixture
def retriever(store, mock_zvec, mock_embedder):
    """Retriever wired to real SQLite store and mocked vector/embed."""
    return Retriever(store, mock_zvec, mock_embedder)


async def _create_peer_and_notes(store, peer_name, contents, **kwargs):
    """Helper: create a peer and insert notes, return (peer, notes)."""
    peer = await store.create_peer(peer_name)
    notes = []
    for content in contents:
        note = await store.create_note(peer_id=peer.id, content=content, **kwargs)
        notes.append(note)
    return peer, notes


# ── TestRetrieverBasic ───────────────────────────────────────────


class TestRetrieverBasic:
    """Basic retrieval behaviour."""

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self, retriever, store):
        peer = await store.create_peer("alice")
        result = await retriever.retrieve("", peer.id)
        assert result == []

    @pytest.mark.asyncio
    async def test_no_notes_returns_empty(self, retriever, store):
        peer = await store.create_peer("bob")
        result = await retriever.retrieve("hello", peer.id)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_retrieval_result_type(self, retriever, store, mock_zvec):
        peer, notes = await _create_peer_and_notes(store, "carol", ["my cat is fluffy"])
        mock_zvec.search.return_value = [{"id": notes[0].id, "score": 0.9}]
        results = await retriever.retrieve("cat", peer.id)
        assert len(results) >= 1
        assert isinstance(results[0], RetrievalResult)

    @pytest.mark.asyncio
    async def test_sorted_by_score_descending(self, retriever, store, mock_zvec):
        peer, notes = await _create_peer_and_notes(
            store, "dave", ["alpha topic notes", "beta topic notes", "gamma topic notes"]
        )
        mock_zvec.search.return_value = [
            {"id": n.id, "score": 0.5} for n in notes
        ]
        results = await retriever.retrieve("topic notes", peer.id)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ── TestRetrieverSourceTracking ──────────────────────────────────


class TestRetrieverSourceTracking:
    """Verify source field is set correctly."""

    @pytest.mark.asyncio
    async def test_fts_only_source(self, retriever, store, mock_zvec):
        peer, notes = await _create_peer_and_notes(store, "eve", ["unique searchterm here"])
        mock_zvec.search.return_value = []  # vector returns nothing
        results = await retriever.retrieve("unique searchterm", peer.id)
        assert len(results) >= 1
        assert results[0].source == "fts"

    @pytest.mark.asyncio
    async def test_vector_only_source(self, retriever, store, mock_zvec):
        peer, notes = await _create_peer_and_notes(store, "frank", ["something about dogs"])
        # FTS won't match "zzz_nomatch", but vector returns the note
        mock_zvec.search.return_value = [{"id": notes[0].id, "score": 0.8}]
        results = await retriever.retrieve("zzz_nomatch", peer.id)
        # May be empty if FTS returns nothing for "zzz_nomatch" — vector provides it
        vec_results = [r for r in results if r.source == "vector"]
        assert len(vec_results) >= 1

    @pytest.mark.asyncio
    async def test_both_source(self, retriever, store, mock_zvec):
        peer, notes = await _create_peer_and_notes(store, "grace", ["my special keyword item"])
        mock_zvec.search.return_value = [{"id": notes[0].id, "score": 0.9}]
        results = await retriever.retrieve("special keyword", peer.id)
        both = [r for r in results if r.source == "both"]
        assert len(both) >= 1


# ── TestRetrieverPeerIsolation ───────────────────────────────────


class TestRetrieverPeerIsolation:
    """Zvec returns cross-peer IDs; they must be filtered out."""

    @pytest.mark.asyncio
    async def test_cross_peer_filtered(self, retriever, store, mock_zvec):
        peer_a, notes_a = await _create_peer_and_notes(store, "hank", ["hank note data"])
        peer_b, notes_b = await _create_peer_and_notes(store, "iris", ["iris note data"])
        # Zvec returns both peers' notes
        mock_zvec.search.return_value = [
            {"id": notes_a[0].id, "score": 0.9},
            {"id": notes_b[0].id, "score": 0.8},
        ]
        results = await retriever.retrieve("note data", peer_a.id)
        returned_peer_ids = {r.note.peer_id for r in results}
        assert peer_b.id not in returned_peer_ids


# ── TestRetrieverGracefulDegradation ─────────────────────────────


class TestRetrieverGracefulDegradation:
    """Search failures degrade gracefully."""

    @pytest.mark.asyncio
    async def test_fts_fails_vector_works(self, retriever, store, mock_zvec):
        peer, notes = await _create_peer_and_notes(store, "jan", ["jan test content"])
        mock_zvec.search.return_value = [{"id": notes[0].id, "score": 0.7}]
        # Sabotage FTS by closing the DB and reopening without FTS table
        # Instead, we mock the db method to raise
        original = retriever._db.fts_search_ranked
        retriever._db.fts_search_ranked = MagicMock(side_effect=Exception("FTS broken"))
        results = await retriever.retrieve("test content", peer.id)
        retriever._db.fts_search_ranked = original
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_zvec_fails_fts_works(self, retriever, store, mock_zvec):
        peer, notes = await _create_peer_and_notes(store, "kim", ["kim findable text"])
        mock_zvec.search.side_effect = Exception("Zvec broken")
        results = await retriever.retrieve("findable text", peer.id)
        assert len(results) >= 1
        assert results[0].source == "fts"

    @pytest.mark.asyncio
    async def test_both_fail_returns_empty(self, retriever, store, mock_zvec):
        peer, notes = await _create_peer_and_notes(store, "leo", ["leo words"])
        mock_zvec.search.side_effect = Exception("Zvec broken")
        retriever._db.fts_search_ranked = MagicMock(side_effect=Exception("FTS broken"))
        results = await retriever.retrieve("words", peer.id)
        assert results == []


# ── TestRetrieverAccessRecording ─────────────────────────────────


class TestRetrieverAccessRecording:
    """Access stats are updated after retrieval."""

    @pytest.mark.asyncio
    async def test_access_count_incremented(self, retriever, store, mock_zvec):
        peer, notes = await _create_peer_and_notes(store, "mike", ["mike record test"])
        mock_zvec.search.return_value = [{"id": notes[0].id, "score": 0.9}]
        await retriever.retrieve("record test", peer.id)
        updated = await store.get_note(notes[0].id)
        assert updated.access_count == 1

    @pytest.mark.asyncio
    async def test_times_surfaced_incremented(self, retriever, store, mock_zvec):
        peer, notes = await _create_peer_and_notes(store, "nina", ["nina surface test"])
        mock_zvec.search.return_value = [{"id": notes[0].id, "score": 0.9}]
        await retriever.retrieve("surface test", peer.id)
        updated = await store.get_note(notes[0].id)
        assert updated.times_surfaced == 1

    @pytest.mark.asyncio
    async def test_last_accessed_at_set(self, retriever, store, mock_zvec):
        peer, notes = await _create_peer_and_notes(store, "omar", ["omar access test"])
        assert notes[0].last_accessed_at is None
        mock_zvec.search.return_value = [{"id": notes[0].id, "score": 0.9}]
        await retriever.retrieve("access test", peer.id)
        updated = await store.get_note(notes[0].id)
        assert updated.last_accessed_at is not None

    @pytest.mark.asyncio
    async def test_multiple_retrievals_accumulate(self, retriever, store, mock_zvec):
        peer, notes = await _create_peer_and_notes(store, "pat", ["pat accum test"])
        mock_zvec.search.return_value = [{"id": notes[0].id, "score": 0.9}]
        await retriever.retrieve("accum test", peer.id)
        await retriever.retrieve("accum test", peer.id)
        updated = await store.get_note(notes[0].id)
        assert updated.access_count == 2
        assert updated.times_surfaced == 2


# ── TestRetrieverLimit ───────────────────────────────────────────


class TestRetrieverLimit:
    """Limit parameter is respected."""

    @pytest.mark.asyncio
    async def test_limit_respected(self, retriever, store, mock_zvec):
        contents = [f"note number {i} content data" for i in range(15)]
        peer, notes = await _create_peer_and_notes(store, "quinn", contents)
        mock_zvec.search.return_value = [
            {"id": n.id, "score": 0.5} for n in notes
        ]
        results = await retriever.retrieve("content data", peer.id, limit=5)
        assert len(results) <= 5


# ── TestRetrieverScoring ─────────────────────────────────────────


class TestRetrieverScoring:
    """All scoring fields are populated."""

    @pytest.mark.asyncio
    async def test_scoring_fields_populated(self, retriever, store, mock_zvec):
        peer, notes = await _create_peer_and_notes(
            store, "rosa", ["rosa scoring test"], importance=0.5
        )
        mock_zvec.search.return_value = [{"id": notes[0].id, "score": 0.9}]
        results = await retriever.retrieve("scoring test", peer.id)
        assert len(results) >= 1
        r = results[0]
        assert r.score > 0
        assert r.rrf_score > 0
        assert r.decay_strength > 0
        assert r.provenance_weight > 0
        assert r.fatigue_factor > 0
        assert r.inference_discount > 0

    @pytest.mark.asyncio
    async def test_inference_discount_applied(self, retriever, store, mock_zvec):
        peer, notes = await _create_peer_and_notes(
            store, "sam", ["sam inference item"], note_type="inference"
        )
        mock_zvec.search.return_value = [{"id": notes[0].id, "score": 0.9}]
        results = await retriever.retrieve("inference item", peer.id)
        assert len(results) >= 1
        assert results[0].inference_discount == 0.7
