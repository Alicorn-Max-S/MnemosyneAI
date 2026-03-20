"""Tests for the static profile generator (Profiler)."""

import pytest

from mnemosyne.config import PROFILE_MIN_NOTES, PROFILE_SECTIONS
from mnemosyne.intelligence.profiler import Profiler
from mnemosyne.pipeline.deriver import Deriver, DeriverAPIError


MOCK_PROFILE_RESPONSE = {
    "identity": "Name is Alice. Lives in Portland, Oregon. Age 32.",
    "professional": "Works as a product manager at Acme Corp.",
    "communication_style": "Prefers concise responses. Likes bullet points.",
    "relationships": "Has a dog named Max. Married to Bob.",
}


@pytest.fixture
async def peer_with_notes(store):
    """Create a peer with enough permanent notes for profile generation."""
    peer = await store.create_peer("Alice")
    for i in range(6):
        await store.create_note(
            peer_id=peer.id,
            content=f"Permanent fact number {i + 1} about Alice.",
            durability="permanent",
            importance=0.9 - i * 0.1,
        )
    return peer


@pytest.fixture
def mock_deriver(monkeypatch):
    """Create a Deriver with mocked _call_api."""
    deriver = Deriver.__new__(Deriver)
    deriver._api_key = "fake-key"
    deriver._base_url = "http://fake"
    deriver._model = "fake-model"
    deriver._client = None
    deriver._headers = {}

    async def fake_call_api(messages, temperature):
        return MOCK_PROFILE_RESPONSE

    monkeypatch.setattr(deriver, "_call_api", fake_call_api)
    return deriver


class TestProfiler:
    """Tests for Profiler."""

    @pytest.mark.asyncio
    async def test_generates_profile(self, store, peer_with_notes, mock_deriver):
        """Mock Deriver → profile has all 4 sections, stored in SQLite."""
        profiler = Profiler(db=store, deriver=mock_deriver)
        profile = await profiler.generate(peer_with_notes.id)

        assert profile is not None
        assert profile.peer_id == peer_with_notes.id
        for section in PROFILE_SECTIONS:
            assert section in profile.sections
            assert profile.sections[section] == MOCK_PROFILE_RESPONSE[section]
        assert profile.fact_count > 0
        assert len(profile.source_note_ids) == 6

        # Verify persisted in SQLite
        stored = await store.get_profile(peer_with_notes.id)
        assert stored is not None
        assert stored.sections == profile.sections

    @pytest.mark.asyncio
    async def test_skips_below_min_notes(self, store, mock_deriver):
        """Peer with < PROFILE_MIN_NOTES permanent notes → returns None."""
        peer = await store.create_peer("Bob")
        for i in range(PROFILE_MIN_NOTES - 1):
            await store.create_note(
                peer_id=peer.id,
                content=f"Fact {i}",
                durability="permanent",
            )
        profiler = Profiler(db=store, deriver=mock_deriver)
        result = await profiler.generate(peer.id)
        assert result is None

    @pytest.mark.asyncio
    async def test_profile_text_format(self, store, peer_with_notes, mock_deriver):
        """Generated text includes section headers and facts."""
        profiler = Profiler(db=store, deriver=mock_deriver)
        await profiler.generate(peer_with_notes.id)

        text = await profiler.get_profile_text(peer_with_notes.id)
        assert text is not None
        assert "## About Alice" in text
        assert "### Identity" in text
        assert "### Professional" in text
        assert "### Communication Style" in text
        assert "### Relationships" in text
        assert "Lives in Portland" in text

    @pytest.mark.asyncio
    async def test_updates_existing_profile(self, store, peer_with_notes, mock_deriver):
        """Second generation replaces the first."""
        profiler = Profiler(db=store, deriver=mock_deriver)
        first = await profiler.generate(peer_with_notes.id)
        assert first is not None

        # Change mock response
        updated_response = {
            "identity": "Name is Alice. Now lives in Seattle.",
            "professional": "Switched to engineering manager.",
            "communication_style": "",
            "relationships": "",
        }

        async def updated_call_api(messages, temperature):
            return updated_response

        mock_deriver._call_api = updated_call_api

        second = await profiler.generate(peer_with_notes.id)
        assert second is not None
        assert second.sections["identity"] == "Name is Alice. Now lives in Seattle."
        assert second.sections["professional"] == "Switched to engineering manager."

        # Only one profile should exist
        stored = await store.get_profile(peer_with_notes.id)
        assert stored.sections["identity"] == second.sections["identity"]

    @pytest.mark.asyncio
    async def test_api_failure_returns_none(self, store, peer_with_notes, monkeypatch):
        """Deriver fails → returns None, no crash."""
        deriver = Deriver.__new__(Deriver)
        deriver._api_key = "fake"
        deriver._base_url = "http://fake"
        deriver._model = "fake"
        deriver._client = None
        deriver._headers = {}

        async def failing_call_api(messages, temperature):
            raise DeriverAPIError("API is down")

        monkeypatch.setattr(deriver, "_call_api", failing_call_api)

        profiler = Profiler(db=store, deriver=deriver)
        result = await profiler.generate(peer_with_notes.id)
        assert result is None
