"""Integration tests for the retrieval pipeline with real Embedder and MemoryAPI."""

import pytest

from mnemosyne.api.memory_api import MemoryAPI
from mnemosyne.models import RetrievalResult
from mnemosyne.vectors.embedder import Embedder


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def embedder():
    """Module-scoped real Embedder (expensive to load)."""
    return Embedder()


@pytest.fixture
async def api(tmp_path, embedder):
    """Function-scoped MemoryAPI with real stores and shared embedder."""
    data_dir = str(tmp_path / "mnemosyne_data")
    mem = MemoryAPI(data_dir=data_dir, embedder=embedder)
    await mem.initialize()
    yield mem
    await mem.close()


# ── Tests ────────────────────────────────────────────────────────


class TestFullRoundTrip:
    """End-to-end: add note, retrieve it with all scoring fields."""

    @pytest.mark.asyncio
    async def test_add_then_retrieve(self, api):
        peer = await api.create_peer("alice")
        await api.add_note(peer.id, "My cat Whiskers loves tuna fish")
        results = await api.retrieve("cat tuna", peer.id)
        assert len(results) >= 1
        r = results[0]
        assert isinstance(r, RetrievalResult)
        assert "tuna" in r.note.content.lower() or "cat" in r.note.content.lower()
        assert r.score > 0
        assert r.rrf_score > 0
        assert r.decay_strength > 0
        assert r.provenance_weight > 0
        assert r.fatigue_factor > 0
        assert r.inference_discount > 0
        assert r.source in ("fts", "vector", "both")


class TestBothSourceContribution:
    """A note found by both FTS and vector should have source='both'."""

    @pytest.mark.asyncio
    async def test_keyword_and_vector_both_contribute(self, api):
        peer = await api.create_peer("bob")
        await api.add_note(peer.id, "quantum entanglement experiments in physics lab")
        results = await api.retrieve("quantum entanglement physics", peer.id)
        assert len(results) >= 1
        # With real embedder + FTS, the note should appear in both channels
        assert results[0].source == "both"


class TestPeerIsolation:
    """Cross-peer notes must not leak."""

    @pytest.mark.asyncio
    async def test_cross_peer_not_returned(self, api):
        alice = await api.create_peer("alice_iso")
        bob = await api.create_peer("bob_iso")
        await api.add_note(alice.id, "alice secret garden memory")
        await api.add_note(bob.id, "bob public park memory")
        results = await api.retrieve("garden memory", bob.id)
        for r in results:
            assert r.note.peer_id == bob.id


class TestScoringRanksRelevant:
    """More relevant notes rank higher."""

    @pytest.mark.asyncio
    async def test_relevant_ranks_above_irrelevant(self, api):
        peer = await api.create_peer("carol")
        await api.add_note(peer.id, "I love my pet golden retriever dog named Buddy")
        await api.add_note(peer.id, "Quantum chromodynamics explains quark interactions")
        results = await api.retrieve("pet dog", peer.id)
        assert len(results) >= 1
        # The pet note should rank first
        assert "dog" in results[0].note.content.lower() or "pet" in results[0].note.content.lower()


class TestAccessRecording:
    """Access stats updated after retrieve."""

    @pytest.mark.asyncio
    async def test_access_count_updated(self, api):
        peer = await api.create_peer("dave")
        note = await api.add_note(peer.id, "dave remembers the blue ocean waves")
        assert note.access_count == 0
        results = await api.retrieve("blue ocean", peer.id)
        assert len(results) >= 1
        updated = await api.get_note(results[0].note.id)
        assert updated.access_count == 1
        assert updated.last_accessed_at is not None


class TestInferenceDiscount:
    """Inference-type notes receive 0.7 discount."""

    @pytest.mark.asyncio
    async def test_inference_discount_applied(self, api):
        peer = await api.create_peer("eve")
        await api.add_note(
            peer.id,
            "eve probably prefers tea over coffee based on context",
            note_type="inference",
        )
        results = await api.retrieve("tea coffee preference", peer.id)
        assert len(results) >= 1
        assert results[0].inference_discount == 0.7


class TestSurfacingFatigue:
    """Repeated retrieval increases fatigue (lower factor)."""

    @pytest.mark.asyncio
    async def test_fatigue_increases_on_repeat(self, api):
        peer = await api.create_peer("frank")
        await api.add_note(peer.id, "frank unique hobby stamp collecting worldwide")
        r1 = await api.retrieve("stamp collecting", peer.id)
        assert len(r1) >= 1
        fatigue_first = r1[0].fatigue_factor

        r2 = await api.retrieve("stamp collecting", peer.id)
        assert len(r2) >= 1
        fatigue_second = r2[0].fatigue_factor
        assert fatigue_second < fatigue_first


class TestLimitRespected:
    """Limit caps results."""

    @pytest.mark.asyncio
    async def test_limit_caps_results(self, api):
        peer = await api.create_peer("grace")
        for i in range(8):
            await api.add_note(peer.id, f"grace memory item number {i} about various topics")
        results = await api.retrieve("memory item topics", peer.id, limit=3)
        assert len(results) <= 3


class TestOldSearchMethodsStillWork:
    """Backward-compat: existing search methods are unbroken."""

    @pytest.mark.asyncio
    async def test_search_keyword_still_works(self, api):
        peer = await api.create_peer("hank")
        await api.add_note(peer.id, "hank likes basketball games")
        results = await api.search_keyword("basketball", peer.id)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_vector_still_works(self, api):
        peer = await api.create_peer("iris")
        await api.add_note(peer.id, "iris enjoys painting landscapes")
        results = await api.search_vector("painting landscapes", peer.id)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_hybrid_still_works(self, api):
        peer = await api.create_peer("jack")
        await api.add_note(peer.id, "jack reads science fiction novels")
        results = await api.search_hybrid("science fiction", peer.id)
        assert len(results) >= 1
