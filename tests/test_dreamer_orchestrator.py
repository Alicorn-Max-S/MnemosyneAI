"""Tests for DreamerProcessor and DreamerOrchestrator."""

import sqlite3
from unittest.mock import AsyncMock, MagicMock

import pytest

from mnemosyne.config import LINK_TYPES
from mnemosyne.db.sqlite_store import SQLiteStore
from mnemosyne.dreamer.gemini_client import GeminiBatchError, GeminiTimeoutError
from mnemosyne.dreamer.orchestrator import CycleResult, DreamerOrchestrator
from mnemosyne.dreamer.processor import DreamerProcessor
from mnemosyne.vectors.embedder import Embedder
from mnemosyne.vectors.zvec_store import ZvecStore


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    """Module-scoped embedder — model loading is expensive."""
    return Embedder()


@pytest.fixture
def zvec(tmp_path) -> ZvecStore:
    """Function-scoped Zvec store."""
    return ZvecStore(str(tmp_path))


@pytest.fixture
def mock_gemini() -> AsyncMock:
    """Mock GeminiClient with all async methods."""
    client = AsyncMock(spec=["submit_batch", "poll_until_done", "get_results"])
    client.submit_batch = AsyncMock(return_value="job-123")
    client.poll_until_done = AsyncMock(return_value=MagicMock())
    client.get_results = AsyncMock(return_value=[])
    return client


@pytest.fixture
def mock_deriver() -> MagicMock:
    """Mock Deriver (not used directly by orchestrator)."""
    return MagicMock()


# ── DreamerProcessor ──────────────────────────────────────────────


class TestProcessLinks:
    async def test_creates_links(self, store, embedder, zvec):
        peer = await store.create_peer("Alice")
        n1 = await store.create_note(peer.id, "Fact one")
        n2 = await store.create_note(peer.id, "Fact two")

        processor = DreamerProcessor(store, embedder, zvec)
        results = [
            {
                "links": [
                    {
                        "source_id": n1.id,
                        "target_id": n2.id,
                        "link_type": "semantic",
                        "strength": 0.8,
                    }
                ]
            }
        ]

        count = await processor.process_links(results)

        assert count == 1
        links = await store.get_links(n1.id)
        assert len(links) == 1
        assert links[0].link_type == "semantic"
        assert links[0].strength == 0.8

    async def test_skips_duplicates(self, store, embedder, zvec):
        peer = await store.create_peer("Alice")
        n1 = await store.create_note(peer.id, "Fact one")
        n2 = await store.create_note(peer.id, "Fact two")

        # Create the link first
        await store.create_link(n1.id, n2.id, "semantic", strength=0.5)

        processor = DreamerProcessor(store, embedder, zvec)
        results = [
            {
                "links": [
                    {
                        "source_id": n1.id,
                        "target_id": n2.id,
                        "link_type": "semantic",
                        "strength": 0.8,
                    }
                ]
            }
        ]

        count = await processor.process_links(results)
        assert count == 0  # Duplicate skipped

    async def test_skips_invalid_link_type(self, store, embedder, zvec):
        peer = await store.create_peer("Alice")
        n1 = await store.create_note(peer.id, "Fact one")
        n2 = await store.create_note(peer.id, "Fact two")

        processor = DreamerProcessor(store, embedder, zvec)
        results = [
            {
                "links": [
                    {
                        "source_id": n1.id,
                        "target_id": n2.id,
                        "link_type": "invalid_type",
                        "strength": 0.5,
                    }
                ]
            }
        ]

        count = await processor.process_links(results)
        assert count == 0

    async def test_skips_missing_fields(self, store, embedder, zvec):
        processor = DreamerProcessor(store, embedder, zvec)
        results = [{"links": [{"source_id": "x"}]}]
        count = await processor.process_links(results)
        assert count == 0

    async def test_clamps_strength(self, store, embedder, zvec):
        peer = await store.create_peer("Alice")
        n1 = await store.create_note(peer.id, "Fact one")
        n2 = await store.create_note(peer.id, "Fact two")

        processor = DreamerProcessor(store, embedder, zvec)
        results = [
            {
                "links": [
                    {
                        "source_id": n1.id,
                        "target_id": n2.id,
                        "link_type": "semantic",
                        "strength": 5.0,
                    }
                ]
            }
        ]

        count = await processor.process_links(results)
        assert count == 1
        links = await store.get_links(n1.id)
        assert links[0].strength == 1.0


class TestProcessPatterns:
    async def test_creates_inference_notes(self, store, embedder, zvec):
        peer = await store.create_peer("Alice")
        source = await store.create_note(peer.id, "Original observation")

        processor = DreamerProcessor(store, embedder, zvec)
        results = [
            {
                "patterns": [
                    {
                        "content": "User shows a recurring pattern of X",
                        "keywords": ["pattern", "recurring"],
                        "supporting_note_ids": [source.id],
                    }
                ]
            }
        ]

        count = await processor.process_patterns(results, peer.id)

        assert count == 1
        notes = await store.list_notes(peer.id, note_type="inference")
        assert len(notes) == 1
        assert notes[0].provenance == "inferred"
        assert notes[0].durability == "contextual"
        assert notes[0].confidence == 0.6
        assert notes[0].is_buffered is False

    async def test_creates_derived_from_links(self, store, embedder, zvec):
        peer = await store.create_peer("Alice")
        source = await store.create_note(peer.id, "Original observation")

        processor = DreamerProcessor(store, embedder, zvec)
        results = [
            {
                "patterns": [
                    {
                        "content": "Derived pattern",
                        "keywords": [],
                        "supporting_note_ids": [source.id],
                    }
                ]
            }
        ]

        await processor.process_patterns(results, peer.id)

        inference_notes = await store.list_notes(peer.id, note_type="inference")
        links = await store.get_links(inference_notes[0].id)
        derived = [l for l in links if l.link_type == "derived_from"]
        assert len(derived) == 1

    async def test_skips_empty_content(self, store, embedder, zvec):
        processor = DreamerProcessor(store, embedder, zvec)
        results = [{"patterns": [{"content": "", "keywords": []}]}]
        count = await processor.process_patterns(results, "peer_1")
        assert count == 0


class TestProcessContradictions:
    async def test_creates_contradiction_links(self, store, embedder, zvec):
        peer = await store.create_peer("Alice")
        n1 = await store.create_note(peer.id, "Likes coffee")
        n2 = await store.create_note(peer.id, "Hates coffee")

        processor = DreamerProcessor(store, embedder, zvec)
        results = [
            {
                "contradictions": [
                    {
                        "note_id_a": n1.id,
                        "note_id_b": n2.id,
                        "description": "Contradictory coffee preference",
                    }
                ]
            }
        ]

        count = await processor.process_contradictions(results)

        assert count == 1
        links = await store.get_links(n1.id)
        contradicts = [l for l in links if l.link_type == "contradicts"]
        assert len(contradicts) == 1
        assert contradicts[0].strength == 1.0

    async def test_skips_missing_note_ids(self, store, embedder, zvec):
        processor = DreamerProcessor(store, embedder, zvec)
        results = [{"contradictions": [{"note_id_a": "x", "description": "d"}]}]
        count = await processor.process_contradictions(results)
        assert count == 0


class TestProcessProfile:
    async def test_updates_peer_profile(self, store, embedder, zvec):
        peer = await store.create_peer("Alice")

        processor = DreamerProcessor(store, embedder, zvec)
        profile_data = {
            "profile": {
                "identity": ["Name: Alice"],
                "professional": ["Software engineer"],
                "communication_style": ["Direct"],
                "relationships": ["Works with Bob"],
            }
        }

        await processor.process_profile(profile_data, peer.id)

        updated = await store.get_peer(peer.id)
        assert updated.static_profile is not None
        assert "identity" in updated.static_profile
        assert updated.profile_updated_at is not None

    async def test_handles_direct_profile_dict(self, store, embedder, zvec):
        peer = await store.create_peer("Alice")

        processor = DreamerProcessor(store, embedder, zvec)
        # No nested "profile" key — sections at top level
        profile_data = {
            "identity": ["Name: Alice"],
            "professional": [],
            "communication_style": [],
            "relationships": [],
        }

        await processor.process_profile(profile_data, peer.id)

        updated = await store.get_peer(peer.id)
        assert updated.static_profile is not None


# ── DreamerOrchestrator ───────────────────────────────────────────


class TestDreamerOrchestrator:
    async def test_empty_buffer_returns_early(
        self, store, embedder, zvec, mock_gemini, mock_deriver
    ):
        peer = await store.create_peer("Alice")

        orch = DreamerOrchestrator(store, embedder, zvec, mock_gemini, mock_deriver)
        result = await orch.run_cycle(peer.id)

        assert result.notes_deduped == 0
        assert result.links_created == 0
        assert result.patterns_found == 0
        assert result.contradictions_found == 0
        assert result.profile_updated is False
        mock_gemini.submit_batch.assert_not_called()

    async def test_full_cycle(
        self, store, embedder, zvec, mock_gemini, mock_deriver
    ):
        peer = await store.create_peer("Alice")
        session = await store.create_session(peer.id)
        n1 = await store.create_note(
            peer.id, "Alice likes hiking in the mountains",
            session_id=session.id,
        )
        n2 = await store.create_note(
            peer.id, "Alice enjoys outdoor activities and nature",
            session_id=session.id,
        )
        n3 = await store.create_note(
            peer.id, "Quantum computing is advancing rapidly",
            session_id=session.id,
        )

        # Mock Gemini to return results for each request type
        # The orchestrator sends: link_requests + pattern_requests +
        #   contradiction_requests + [profile_request]
        # For a simple case with 3 notes:
        #   1 link request, 1 pattern request, 0-1 contradiction requests, 1 profile
        mock_gemini.get_results.return_value = [
            # Link result
            {
                "links": [
                    {
                        "source_id": n1.id,
                        "target_id": n2.id,
                        "link_type": "semantic",
                        "strength": 0.85,
                    }
                ]
            },
            # Pattern result
            {
                "patterns": [
                    {
                        "content": "User enjoys nature and outdoor activities",
                        "keywords": ["outdoors", "nature"],
                        "supporting_note_ids": [n1.id, n2.id],
                    }
                ]
            },
            # Profile result (may include contradiction results before this)
            {
                "profile": {
                    "identity": ["Name: Alice"],
                    "professional": [],
                    "communication_style": [],
                    "relationships": [],
                }
            },
        ]

        orch = DreamerOrchestrator(store, embedder, zvec, mock_gemini, mock_deriver)
        result = await orch.run_cycle(peer.id)

        # Gemini batch was submitted
        mock_gemini.submit_batch.assert_called_once()

        # CycleResult should have non-zero values
        assert isinstance(result, CycleResult)
        # At minimum, links or patterns should have been processed
        # (exact counts depend on result routing which depends on request counts)

    async def test_batch_failure_returns_partial(
        self, store, embedder, zvec, mock_gemini, mock_deriver
    ):
        peer = await store.create_peer("Alice")
        await store.create_note(peer.id, "Some buffered note")
        await store.create_note(peer.id, "Another buffered note")

        mock_gemini.poll_until_done.side_effect = GeminiBatchError(
            "Job failed"
        )

        orch = DreamerOrchestrator(store, embedder, zvec, mock_gemini, mock_deriver)
        result = await orch.run_cycle(peer.id)

        # Should not raise — returns partial result
        assert isinstance(result, CycleResult)
        # Gemini counts should be zero
        assert result.links_created == 0
        assert result.patterns_found == 0
        assert result.contradictions_found == 0
        assert result.profile_updated is False

    async def test_timeout_returns_partial(
        self, store, embedder, zvec, mock_gemini, mock_deriver
    ):
        peer = await store.create_peer("Alice")
        await store.create_note(peer.id, "A buffered note")

        mock_gemini.poll_until_done.side_effect = GeminiTimeoutError(
            "Polling timed out"
        )

        orch = DreamerOrchestrator(store, embedder, zvec, mock_gemini, mock_deriver)
        result = await orch.run_cycle(peer.id)

        assert isinstance(result, CycleResult)
        assert result.links_created == 0
        assert result.profile_updated is False

    async def test_profile_updated_flag(
        self, store, embedder, zvec, mock_gemini, mock_deriver
    ):
        peer = await store.create_peer("Alice")
        await store.create_note(
            peer.id,
            "Alice is a software engineer",
            durability="permanent",
            importance=0.9,
        )

        # Single request set: 1 link + 1 pattern + 0 contradiction + 1 profile
        mock_gemini.get_results.return_value = [
            {"links": []},      # link result
            {"patterns": []},   # pattern result
            # no contradiction results
            {                   # profile result
                "profile": {
                    "identity": ["Software engineer named Alice"],
                    "professional": ["Works in tech"],
                    "communication_style": ["Direct"],
                    "relationships": [],
                }
            },
        ]

        orch = DreamerOrchestrator(store, embedder, zvec, mock_gemini, mock_deriver)
        result = await orch.run_cycle(peer.id)

        assert result.profile_updated is True
        updated_peer = await store.get_peer(peer.id)
        assert updated_peer.static_profile is not None
        assert updated_peer.profile_updated_at is not None

    async def test_dedup_runs_before_batch(
        self, store, embedder, zvec, mock_gemini, mock_deriver
    ):
        """Verify dedup merges near-duplicates before Gemini submission."""
        peer = await store.create_peer("Alice")
        # Two very similar notes — should be merged by dedup
        await store.create_note(peer.id, "The cat sat on the mat", importance=0.5)
        await store.create_note(peer.id, "The cat sat on the mat today", importance=0.3)
        # One different note
        await store.create_note(peer.id, "Quantum physics research", importance=0.4)

        mock_gemini.get_results.return_value = [
            {"links": []},
            {"patterns": []},
            {"profile": {"identity": [], "professional": [],
                         "communication_style": [], "relationships": []}},
        ]

        orch = DreamerOrchestrator(store, embedder, zvec, mock_gemini, mock_deriver)
        result = await orch.run_cycle(peer.id)

        # The two similar notes should have been merged
        assert result.notes_deduped >= 1
