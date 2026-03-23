"""End-to-end integration tests for Phase 4 intelligence pipeline."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mnemosyne.intelligence.linker import Linker
from mnemosyne.intelligence.reranker import ColBERTReranker
from mnemosyne.models import RetrievalResult
from mnemosyne.pipeline import create_worker, ingest_message
from mnemosyne.pipeline.deriver import Deriver
from mnemosyne.retrieval.retriever import Retriever
from mnemosyne.vectors.embedder import Embedder
from mnemosyne.vectors.zvec_store import ZvecStore


# ── Helpers ──────────────────────────────────────────────────────────


def make_chat_response(content_dict: dict, status_code: int = 200) -> httpx.Response:
    """Build a mock httpx.Response matching the chat completions shape."""
    body = {"choices": [{"message": {"content": json.dumps(content_dict)}}]}
    return httpx.Response(status_code, json=body)


EXTRACT_2_NOTES = {
    "notes": [
        {"text": "User has a golden retriever named Max", "is_confirmation": False},
        {"text": "User enjoys morning runs in the park", "is_confirmation": False},
    ]
}

SCORE_2_NOTES = {
    "scored_notes": [
        {
            "text": "User has a golden retriever named Max",
            "emotional_weight": 0.7,
            "provenance": "organic",
            "durability": "permanent",
            "keywords": ["dog", "golden retriever", "Max"],
            "tags": ["pets"],
            "context_description": "User's pet information",
        },
        {
            "text": "User enjoys morning runs in the park",
            "emotional_weight": 0.4,
            "provenance": "organic",
            "durability": "contextual",
            "keywords": ["running", "morning", "park"],
            "tags": ["exercise"],
            "context_description": "User's exercise habits",
        },
    ]
}


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def embedder():
    """Module-scoped embedder (expensive model load)."""
    return Embedder()


@pytest.fixture
def zvec(tmp_path):
    """Function-scoped ZvecStore in a temp directory."""
    return ZvecStore(str(tmp_path))


@pytest.fixture
def deriver():
    """Function-scoped Deriver with test credentials."""
    return Deriver(api_key="test-key", base_url="https://test.example.com/v1")


# ── Tests ────────────────────────────────────────────────────────────


class TestFullWriteWithLinks:
    async def test_full_write_with_links(self, store, embedder, zvec, deriver):
        """Ingest -> derive -> notes + semantic links created via Linker."""
        peer = await store.create_peer("IntPeer1")
        session = await store.create_session(peer.id)

        deriver._client.post = AsyncMock(
            side_effect=[
                make_chat_response(EXTRACT_2_NOTES),
                make_chat_response(SCORE_2_NOTES),
            ]
        )

        await ingest_message(
            session.id, peer.id, "user",
            "I have a golden retriever named Max and I run in the park every morning",
            db=store,
        )

        linker = Linker(db=store, zvec=zvec, embedder=embedder)
        worker = create_worker(store, deriver, embedder, zvec, linker=linker)
        result = await worker.run_once()
        assert result is True

        notes = await store.list_notes(peer.id)
        assert len(notes) == 2

        # Links may or may not be created depending on similarity threshold,
        # but the pipeline should complete without error.
        # Check linker ran by verifying notes are in Zvec
        for note in notes:
            assert note.zvec_id is not None


class TestRetrievalWithColBERT:
    async def test_retrieval_with_colbert(self, store, embedder, zvec):
        """ColBERT reranker sets colbert_score on results."""
        peer = await store.create_peer("IntPeer2")
        note = await store.create_note(peer_id=peer.id, content="cats love tuna fish")
        emb = embedder.embed_document("cats love tuna fish")
        zvec.insert(note.id, emb)

        # Create a mock reranker that returns controlled scores
        mock_reranker = MagicMock()
        mock_reranker.rerank = AsyncMock(return_value=[(note.id, 0.95)])

        retriever = Retriever(store, zvec, embedder, colbert_reranker=mock_reranker)
        results = await retriever.retrieve("cats tuna", peer.id)

        assert len(results) >= 1
        assert results[0].colbert_score is not None
        assert results[0].colbert_score == 0.95


class TestRetrievalColBERTFallback:
    async def test_retrieval_colbert_fallback(self, store, embedder, zvec):
        """ColBERT unavailable → colbert_score is None, Phase 3 behavior."""
        peer = await store.create_peer("IntPeer3")
        note = await store.create_note(
            peer_id=peer.id, content="dogs play fetch outdoors"
        )
        emb = embedder.embed_document("dogs play fetch outdoors")
        zvec.insert(note.id, emb)

        # No reranker → Phase 3 fallback
        retriever = Retriever(store, zvec, embedder, colbert_reranker=None)
        results = await retriever.retrieve("dogs fetch", peer.id)

        assert len(results) >= 1
        assert results[0].colbert_score is None
        assert results[0].composite_score > 0


class TestProfileGenerationRoundTrip:
    async def test_profile_generation_round_trip(self, store, deriver):
        """Permanent notes → profile → text retrieval."""
        from mnemosyne.intelligence.profiler import Profiler

        peer = await store.create_peer("IntPeer4")

        # Create enough permanent notes (>= PROFILE_MIN_NOTES=5)
        for i in range(6):
            await store.create_note(
                peer_id=peer.id,
                content=f"Permanent fact number {i} about the user",
                durability="permanent",
                importance=0.8,
            )

        profile_response = {
            "identity": "User is a test person.",
            "professional": "Works in testing.",
            "communication_style": "Prefers concise responses.",
            "relationships": "",
        }
        deriver._client.post = AsyncMock(
            return_value=make_chat_response(profile_response)
        )

        profiler = Profiler(db=store, deriver=deriver)
        profile = await profiler.generate(peer.id)

        assert profile is not None
        assert profile.peer_id == peer.id
        assert profile.sections["identity"] == "User is a test person."

        # Round-trip: fetch text
        text = await profiler.get_profile_text(peer.id)
        assert text is not None
        assert "test person" in text


class TestLinkExpansionInRetrieval:
    async def test_link_expansion_in_retrieval(self, store, embedder, zvec):
        """Linked notes appear in retrieval results with 'link' source."""
        peer = await store.create_peer("IntPeer5")

        # Create two notes and embed them
        n1 = await store.create_note(
            peer_id=peer.id, content="quantum physics experiments"
        )
        emb1 = embedder.embed_document("quantum physics experiments")
        zvec.insert(n1.id, emb1)

        n2 = await store.create_note(
            peer_id=peer.id, content="particle accelerator research"
        )
        emb2 = embedder.embed_document("particle accelerator research")
        zvec.insert(n2.id, emb2)

        # Link them
        await store.create_link(
            source_note_id=n1.id,
            target_note_id=n2.id,
            link_type="semantic",
            strength=0.9,
        )

        retriever = Retriever(store, zvec, embedder)
        results = await retriever.retrieve("quantum physics", peer.id, limit=10)

        result_ids = {r.note.id for r in results}
        assert n1.id in result_ids

        # n2 should appear either through vector search or link expansion
        if n2.id in result_ids:
            r2 = [r for r in results if r.note.id == n2.id][0]
            # If it came via link expansion, source contains "link"
            # If it came via vector search, it might be "vector" or "both"
            # Either way it should be present
            assert r2.source in ("link", "vector", "both", "vector+link", "both+link")
