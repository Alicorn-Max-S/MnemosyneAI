"""End-to-end integration tests for the Dreamer pipeline.

Buffer notes → dedup → dream → verify links/patterns/profile/entities.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mnemosyne.config import COLBERT_TOKEN_DIM
from mnemosyne.dreamer.orchestrator import DreamerOrchestrator
from mnemosyne.graph import create_magma_graph
from mnemosyne.intelligence.reranker import ColBERTReranker
from mnemosyne.pipeline import create_worker, ingest_message
from mnemosyne.pipeline.deriver import Deriver
from mnemosyne.vectors.embedder import Embedder
from mnemosyne.vectors.zvec_store import ZvecStore


# ── Helpers ──────────────────────────────────────────────────────────


def make_chat_response(content_dict: dict, status_code: int = 200) -> httpx.Response:
    """Build a mock httpx.Response matching the chat completions shape."""
    body = {"choices": [{"message": {"content": json.dumps(content_dict)}}]}
    return httpx.Response(status_code, json=body)


def make_extract_response(text: str) -> dict:
    """Build extraction response with a single note."""
    return {"notes": [{"text": text, "is_confirmation": False}]}


def make_score_response(text: str, **overrides) -> dict:
    """Build scoring response with a single scored note."""
    note = {
        "text": text,
        "emotional_weight": 0.6,
        "provenance": "organic",
        "durability": "permanent",
        "keywords": [],
        "tags": [],
        "context_description": "Test context",
    }
    note.update(overrides)
    return {"scored_notes": [note]}


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def embedder():
    """Module-scoped embedder (expensive model load)."""
    return Embedder()


@pytest.fixture(scope="module")
def colbert_reranker():
    """Module-scoped ColBERT reranker (expensive model load)."""
    r = ColBERTReranker()
    if not r._ensure_loaded():
        pytest.skip("ColBERT model not available")
    return r


@pytest.fixture
def zvec(tmp_path):
    """Function-scoped ZvecStore in a temp directory."""
    return ZvecStore(str(tmp_path))


@pytest.fixture
def deriver():
    """Function-scoped Deriver with test credentials."""
    return Deriver(api_key="test-key", base_url="https://test.example.com/v1")


@pytest.fixture
def mock_gemini():
    """Mock GeminiClient with all async methods."""
    client = AsyncMock(spec=["submit_batch", "poll_until_done", "get_results"])
    client.submit_batch = AsyncMock(return_value="job-123")
    client.poll_until_done = AsyncMock(return_value=MagicMock())
    client.get_results = AsyncMock(return_value=[])
    return client


# ── Test: Full Pipeline ──────────────────────────────────────────────


MESSAGES = [
    "I work at Google as a software engineer",
    "My dog Buddy is a golden retriever",
    "John Smith is my manager at Google",
    "I love hiking in the Rocky Mountains on weekends",
    "Sarah Johnson is my best friend from college",
    "I play guitar in a band called The Wavelengths",
    "My wife Emily is a doctor at Stanford Hospital",
    "I grew up in Portland Oregon and moved to San Francisco",
    "I have a meeting with John Smith every Monday morning",
    "My favorite programming language is Python",
]


class TestDreamerPipeline:
    async def test_full_pipeline(
        self, store, embedder, zvec, colbert_reranker, deriver, mock_gemini
    ):
        """End-to-end: ingest → worker → verify tokens/entities → dreamer → verify results."""
        peer = await store.create_peer("PipelinePeer")
        session = await store.create_session(peer.id)

        magma_graph = create_magma_graph(store)

        # Ingest and process each message
        for msg in MESSAGES:
            extract = make_extract_response(msg)
            score = make_score_response(msg)
            deriver._client.post = AsyncMock(
                side_effect=[
                    make_chat_response(extract),
                    make_chat_response(score),
                ]
            )

            await ingest_message(session.id, peer.id, "user", msg, db=store)
            worker = create_worker(
                store, deriver, embedder, zvec,
                colbert_reranker=colbert_reranker,
                magma_graph=magma_graph,
            )
            await worker.run_once()

        # Verify buffered notes exist
        notes = await store.list_notes(peer.id)
        assert len(notes) == len(MESSAGES)
        assert all(n.is_buffered for n in notes)

        # Verify ColBERT tokens stored for each note
        note_ids = [n.id for n in notes]
        tokens = await store.get_colbert_tokens(note_ids)
        assert len(tokens) == len(notes)
        # Verify token blobs have correct shape
        for nid, blob in tokens.items():
            import numpy as np
            arr = np.frombuffer(blob, dtype=np.float32)
            assert arr.size % COLBERT_TOKEN_DIM == 0

        # Verify entity mentions created by MAGMA during handle_derive
        entities = await store.get_entities_for_peer(peer.id)
        # Messages contain capitalized names like "John Smith", "Google", etc.
        entity_names = [e["entity_name"] if isinstance(e, dict) else e[0] for e in entities]
        assert len(entity_names) > 0  # at least some entities found

        # Set up mock Gemini to return link + pattern + contradiction + profile results
        note_a = notes[0]
        note_b = notes[1]
        gemini_results = [
            # Link result
            {
                "links": [
                    {
                        "source_id": note_a.id,
                        "target_id": note_b.id,
                        "link_type": "semantic",
                        "strength": 0.85,
                    }
                ]
            },
            # Pattern result
            {
                "patterns": [
                    {
                        "content": "User has strong ties to Google",
                        "keywords": ["google", "work"],
                        "supporting_note_ids": [note_a.id],
                    }
                ]
            },
            # Contradiction result (empty — no real contradictions)
            {"contradictions": []},
            # Profile result
            {
                "profile": {
                    "identity": ["Software engineer", "Dog owner"],
                    "professional": ["Works at Google"],
                    "communication_style": ["Direct and technical"],
                    "relationships": ["Manager: John Smith"],
                }
            },
        ]
        mock_gemini.get_results = AsyncMock(return_value=gemini_results)

        # Run dreamer cycle
        orch = DreamerOrchestrator(
            store, embedder, zvec, mock_gemini, MagicMock()
        )
        result = await orch.run_cycle(peer.id)

        # Verify dreamer completed
        assert result.profile_updated is True

        # Verify profile stored on peer
        peer_after = await store.get_peer(peer.id)
        assert peer_after.static_profile is not None

    async def test_colbert_tokens_stored_per_note(
        self, store, embedder, zvec, colbert_reranker, deriver
    ):
        """ColBERT token embeddings stored in colbert_tokens table for each derived note."""
        peer = await store.create_peer("ColBERTTokenPeer")
        session = await store.create_session(peer.id)

        text = "Testing ColBERT token storage"
        deriver._client.post = AsyncMock(
            side_effect=[
                make_chat_response(make_extract_response(text)),
                make_chat_response(make_score_response(text)),
            ]
        )

        await ingest_message(session.id, peer.id, "user", text, db=store)
        worker = create_worker(
            store, deriver, embedder, zvec,
            colbert_reranker=colbert_reranker,
        )
        await worker.run_once()

        notes = await store.list_notes(peer.id)
        assert len(notes) == 1
        tokens = await store.get_colbert_tokens([notes[0].id])
        assert notes[0].id in tokens

    async def test_empty_buffer_returns_early(
        self, store, embedder, zvec, mock_gemini
    ):
        """Dreamer cycle with no buffered notes returns immediately."""
        peer = await store.create_peer("EmptyBufferPeer")

        orch = DreamerOrchestrator(
            store, embedder, zvec, mock_gemini, MagicMock()
        )
        result = await orch.run_cycle(peer.id)

        assert result.notes_deduped == 0
        assert result.links_created == 0
        mock_gemini.submit_batch.assert_not_called()

    async def test_rerun_after_cycle_returns_early(
        self, store, embedder, zvec, colbert_reranker, deriver, mock_gemini
    ):
        """After dedup merges similar notes, re-running returns early."""
        peer = await store.create_peer("RerunPeer")
        session = await store.create_session(peer.id)

        # Use 2 similar messages so dedup will cluster and merge them,
        # clearing is_buffered=0 on all cluster members.
        texts = [
            "My dog Buddy is a golden retriever who loves swimming",
            "My golden retriever Buddy enjoys swimming a lot",
        ]
        for text in texts:
            deriver._client.post = AsyncMock(
                side_effect=[
                    make_chat_response(make_extract_response(text)),
                    make_chat_response(make_score_response(text)),
                ]
            )
            await ingest_message(session.id, peer.id, "user", text, db=store)
            worker = create_worker(
                store, deriver, embedder, zvec,
                colbert_reranker=colbert_reranker,
            )
            await worker.run_once()

        orch = DreamerOrchestrator(
            store, embedder, zvec, mock_gemini, MagicMock()
        )

        # First cycle dedup-merges the similar notes and processes batch
        await orch.run_cycle(peer.id)

        # Reset mock to track second call
        mock_gemini.submit_batch.reset_mock()

        # Second cycle — buffer cleared by dedup merge
        result2 = await orch.run_cycle(peer.id)
        assert result2.notes_deduped == 0
        mock_gemini.submit_batch.assert_not_called()


class TestEntityExtractionIntegration:
    async def test_entities_extracted_during_derive(
        self, store, embedder, zvec, colbert_reranker, deriver
    ):
        """Ingesting a message mentioning entities → entity_mentions table populated."""
        peer = await store.create_peer("EntityPeer")
        session = await store.create_session(peer.id)

        # Entity must NOT be at sentence start (extractor skips position 0)
        # and must be multi-word capitalized (extractor regex requires 2+ words)
        text = "I work with John Smith at the office every day"
        deriver._client.post = AsyncMock(
            side_effect=[
                make_chat_response(make_extract_response(text)),
                make_chat_response(make_score_response(text)),
            ]
        )

        magma_graph = create_magma_graph(store)

        await ingest_message(session.id, peer.id, "user", text, db=store)
        worker = create_worker(
            store, deriver, embedder, zvec,
            colbert_reranker=colbert_reranker,
            magma_graph=magma_graph,
        )
        await worker.run_once()

        # Verify entity mentions in SQLite
        entities = await store.get_entities_for_peer(peer.id)
        entity_names = [e["entity_name"] if isinstance(e, dict) else e[0] for e in entities]
        # MAGMA's rule-based extraction should find "John Smith"
        assert len(entity_names) > 0
        assert any("john smith" in name.lower() for name in entity_names)
