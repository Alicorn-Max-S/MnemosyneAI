"""Integration tests for the full write pipeline: intake -> worker -> derive handler."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from mnemosyne.pipeline import create_worker, ingest_message
from mnemosyne.pipeline.deriver import Deriver
from mnemosyne.vectors.embedder import Embedder
from mnemosyne.vectors.zvec_store import ZvecStore


# ── Helpers ──────────────────────────────────────────────────────────


def make_chat_response(content_dict: dict, status_code: int = 200) -> httpx.Response:
    """Build a mock httpx.Response matching the chat completions shape."""
    body = {"choices": [{"message": {"content": json.dumps(content_dict)}}]}
    return httpx.Response(status_code, json=body)


def make_error_response(status_code: int) -> httpx.Response:
    """Build a mock httpx.Response with the given error status code."""
    return httpx.Response(status_code, json={"error": "mock error"})


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


# ── Mock data ────────────────────────────────────────────────────────

EXTRACT_2_NOTES = {
    "notes": [
        {"text": "User has a golden retriever named Buddy", "is_confirmation": False},
        {"text": "User goes hiking on weekends", "is_confirmation": False},
    ]
}

SCORE_2_NOTES = {
    "scored_notes": [
        {
            "text": "User has a golden retriever named Buddy",
            "emotional_weight": 0.7,
            "provenance": "organic",
            "durability": "permanent",
            "keywords": ["dog", "golden retriever", "Buddy"],
            "tags": ["pets"],
            "context_description": "User's pet information",
        },
        {
            "text": "User goes hiking on weekends",
            "emotional_weight": 0.4,
            "provenance": "organic",
            "durability": "contextual",
            "keywords": ["hiking", "weekends"],
            "tags": ["hobbies"],
            "context_description": "User's weekend activities",
        },
    ]
}


# ── Tests ────────────────────────────────────────────────────────────


class TestWritePipeline:
    async def test_full_round_trip(self, store, embedder, zvec, deriver):
        """Full pipeline: ingest -> derive -> 2 notes in SQLite + Zvec, FTS5 works."""
        peer = await store.create_peer("Alice")
        session = await store.create_session(peer.id)

        # Mock extract then score
        deriver._client.post = AsyncMock(
            side_effect=[
                make_chat_response(EXTRACT_2_NOTES),
                make_chat_response(SCORE_2_NOTES),
            ]
        )

        await ingest_message(session.id, peer.id, "user",
                             "I have a golden retriever named Buddy and I hike on weekends",
                             db=store)

        worker = create_worker(store, deriver, embedder, zvec)
        result = await worker.run_once()
        assert result is True

        # 2 notes stored in SQLite
        notes = await store.list_notes(peer.id)
        assert len(notes) == 2
        assert all(n.is_buffered for n in notes)

        # FTS5 finds "golden retriever"
        fts_results = await store.fts_search("golden retriever", peer.id)
        assert len(fts_results) >= 1
        assert any("golden retriever" in n.content for n, _ in fts_results)

        # Zvec finds the Buddy note via vector search
        query_emb = embedder.embed_query("dog named Buddy")
        vec_results = zvec.search(query_emb, top_k=5)
        assert len(vec_results) >= 1

        # Task completed — no more derive tasks
        task = await store.dequeue_task("derive")
        assert task is None

    async def test_429_retry_then_success(self, store, embedder, zvec, deriver):
        """429 on first call, then extract + score succeed. Notes stored."""
        peer = await store.create_peer("Bob")
        session = await store.create_session(peer.id)

        deriver._client.post = AsyncMock(
            side_effect=[
                make_error_response(429),
                make_chat_response(EXTRACT_2_NOTES),
                make_chat_response(SCORE_2_NOTES),
            ]
        )

        await ingest_message(session.id, peer.id, "user", "My dog Buddy", db=store)

        with patch("mnemosyne.pipeline.deriver.asyncio.sleep", new_callable=AsyncMock):
            worker = create_worker(store, deriver, embedder, zvec)
            await worker.run_once()

        notes = await store.list_notes(peer.id)
        assert len(notes) == 2
        assert deriver._client.post.call_count == 3

    async def test_zvec_failure_graceful(self, store, embedder, zvec, deriver):
        """Zvec insert fails per-note — notes still in SQLite, zvec_id is None."""
        peer = await store.create_peer("Carol")
        session = await store.create_session(peer.id)

        deriver._client.post = AsyncMock(
            side_effect=[
                make_chat_response(EXTRACT_2_NOTES),
                make_chat_response(SCORE_2_NOTES),
            ]
        )

        # Monkey-patch zvec.insert_batch to always raise (handler uses batch insert)
        original_insert_batch = zvec.insert_batch
        zvec.insert_batch = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("zvec boom"))

        await ingest_message(session.id, peer.id, "user", "Buddy the dog", db=store)
        worker = create_worker(store, deriver, embedder, zvec)
        await worker.run_once()

        zvec.insert_batch = original_insert_batch

        notes = await store.list_notes(peer.id)
        assert len(notes) == 2
        assert all(n.zvec_id is None for n in notes)

    async def test_empty_extraction_completes(self, store, embedder, zvec, deriver):
        """Extract returns no notes — task still completes, 0 notes stored."""
        peer = await store.create_peer("Dave")
        session = await store.create_session(peer.id)

        deriver._client.post = AsyncMock(
            return_value=make_chat_response({"notes": []})
        )

        await ingest_message(session.id, peer.id, "user", "hello", db=store)
        worker = create_worker(store, deriver, embedder, zvec)
        result = await worker.run_once()

        assert result is True
        notes = await store.list_notes(peer.id)
        assert len(notes) == 0

        # Task completed
        task = await store.dequeue_task("derive")
        assert task is None

    async def test_assistant_no_derive(self, store, embedder, zvec, deriver):
        """Assistant messages don't enqueue derive tasks."""
        peer = await store.create_peer("Eve")
        session = await store.create_session(peer.id)

        await ingest_message(session.id, peer.id, "assistant", "Hi there!", db=store)

        worker = create_worker(store, deriver, embedder, zvec)
        result = await worker.run_once()
        assert result is False
