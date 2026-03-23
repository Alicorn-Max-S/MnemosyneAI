"""Tests for Gemini client, Dreamer prompts, and task builder."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mnemosyne.config import (
    DREAMER_CONTRADICTION_TEMPERATURE,
    DREAMER_LINK_TEMPERATURE,
    DREAMER_PATTERN_TEMPERATURE,
    DREAMER_PROFILE_TEMPERATURE,
    GEMINI_MODEL,
    PROFILE_SECTIONS,
)
from mnemosyne.dreamer.gemini_client import (
    GeminiBatchError,
    GeminiClient,
    GeminiTimeoutError,
)
from mnemosyne.dreamer.prompts import (
    CONTRADICTION_DETECTION_PROMPT,
    LINK_GENERATION_PROMPT,
    PATTERN_DETECTION_PROMPT,
    PROFILE_UPDATE_PROMPT,
)
from mnemosyne.dreamer.task_builder import (
    _CONTRADICTION_BATCH_SIZE,
    _LINK_BATCH_SIZE,
    build_contradiction_requests,
    build_link_requests,
    build_pattern_requests,
    build_profile_request,
)
from mnemosyne.models import Link, Note, Session


# ── Helpers ───────────────────────────────────────────────────────


def _make_note(note_id: str = "note_1", peer_id: str = "peer_1",
               content: str = "Some fact", **kwargs) -> Note:
    """Create a minimal Note for testing."""
    defaults = {
        "id": note_id,
        "peer_id": peer_id,
        "content": content,
        "created_at": "2026-01-01T00:00:00.000Z",
        "updated_at": "2026-01-01T00:00:00.000Z",
    }
    defaults.update(kwargs)
    return Note(**defaults)


def _make_link(source: str = "n1", target: str = "n2",
               link_type: str = "semantic") -> Link:
    """Create a minimal Link for testing."""
    return Link(
        id="link_1",
        source_note_id=source,
        target_note_id=target,
        link_type=link_type,
        created_at="2026-01-01T00:00:00.000Z",
    )


def _make_session(session_id: str = "sess_1", peer_id: str = "peer_1") -> Session:
    """Create a minimal Session for testing."""
    return Session(
        id=session_id,
        peer_id=peer_id,
        started_at="2026-01-01T00:00:00.000Z",
    )


def _mock_state(name: str) -> MagicMock:
    """Create a mock job state with a .name attribute."""
    state = MagicMock()
    state.name = name
    return state


def _mock_job(name: str = "batches/123", state: str = "JOB_STATE_SUCCEEDED") -> MagicMock:
    """Create a mock batch job object."""
    job = MagicMock()
    job.name = name
    job.state = _mock_state(state)
    return job


def _mock_response(text: str) -> MagicMock:
    """Create a mock inline response with text content."""
    resp = MagicMock()
    part = MagicMock()
    part.text = text
    content = MagicMock()
    content.parts = [part]
    candidate = MagicMock()
    candidate.content = content
    resp.candidates = [candidate]
    return resp


@pytest.fixture
def gemini_client():
    """GeminiClient with fully mocked google.genai.Client."""
    mock_sdk = MagicMock()
    with patch("mnemosyne.dreamer.gemini_client.genai.Client", return_value=mock_sdk):
        client = GeminiClient(api_key="test-key")
    return client, mock_sdk


# ── GeminiClient Tests ────────────────────────────────────────────


class TestGeminiClientSubmit:
    async def test_submit_batch_returns_job_name(self, gemini_client):
        client, mock_sdk = gemini_client
        mock_sdk.batches.create.return_value = _mock_job("batches/abc")

        result = await client.submit_batch([{"test": "req"}], "test-batch")

        assert result == "batches/abc"

    async def test_submit_batch_passes_correct_args(self, gemini_client):
        client, mock_sdk = gemini_client
        mock_sdk.batches.create.return_value = _mock_job()
        requests = [{"contents": "test"}]

        await client.submit_batch(requests, "my-batch")

        mock_sdk.batches.create.assert_called_once_with(
            model=GEMINI_MODEL,
            src=requests,
            config={"display_name": "my-batch"},
        )


class TestGeminiClientPoll:
    async def test_poll_succeeds_immediately(self, gemini_client):
        client, mock_sdk = gemini_client
        job = _mock_job(state="JOB_STATE_SUCCEEDED")
        mock_sdk.batches.get.return_value = job

        result = await client.poll_until_done("batches/123", poll_interval=0.01)

        assert result is job

    async def test_poll_waits_then_succeeds(self, gemini_client):
        client, mock_sdk = gemini_client
        running_job = _mock_job(state="JOB_STATE_RUNNING")
        done_job = _mock_job(state="JOB_STATE_SUCCEEDED")
        mock_sdk.batches.get.side_effect = [running_job, done_job]

        with patch("mnemosyne.dreamer.gemini_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await client.poll_until_done("batches/123", poll_interval=5.0)

        assert result is done_job
        mock_sleep.assert_called_once_with(5.0)

    async def test_poll_raises_on_failed(self, gemini_client):
        client, mock_sdk = gemini_client
        mock_sdk.batches.get.return_value = _mock_job(state="JOB_STATE_FAILED")

        with pytest.raises(GeminiBatchError, match="FAILED"):
            await client.poll_until_done("batches/123", poll_interval=0.01)

    async def test_poll_raises_on_cancelled(self, gemini_client):
        client, mock_sdk = gemini_client
        mock_sdk.batches.get.return_value = _mock_job(state="JOB_STATE_CANCELLED")

        with pytest.raises(GeminiBatchError, match="CANCELLED"):
            await client.poll_until_done("batches/123", poll_interval=0.01)

    async def test_poll_raises_on_expired(self, gemini_client):
        client, mock_sdk = gemini_client
        mock_sdk.batches.get.return_value = _mock_job(state="JOB_STATE_EXPIRED")

        with pytest.raises(GeminiBatchError, match="EXPIRED"):
            await client.poll_until_done("batches/123", poll_interval=0.01)

    async def test_poll_raises_on_timeout(self, gemini_client):
        client, mock_sdk = gemini_client
        mock_sdk.batches.get.return_value = _mock_job(state="JOB_STATE_RUNNING")

        # max_time=0 ensures the first elapsed check triggers timeout
        with patch("mnemosyne.dreamer.gemini_client.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(GeminiTimeoutError, match="still in state"):
                await client.poll_until_done("batches/123", poll_interval=0.01, max_time=0.0)


class TestGeminiClientResults:
    async def test_get_results_parses_inline(self, gemini_client):
        client, _ = gemini_client
        job = MagicMock()
        job.dest.inlined_responses = [
            _mock_response('{"links": []}'),
            _mock_response('{"patterns": [{"content": "test"}]}'),
        ]

        results = await client.get_results(job)

        assert len(results) == 2
        assert results[0] == {"links": []}
        assert results[1] == {"patterns": [{"content": "test"}]}

    async def test_get_results_skips_unparseable(self, gemini_client):
        client, _ = gemini_client
        job = MagicMock()
        job.dest.inlined_responses = [
            _mock_response('{"valid": true}'),
            _mock_response("not json at all"),
            _mock_response('{"also_valid": true}'),
        ]

        results = await client.get_results(job)

        assert len(results) == 2
        assert results[0] == {"valid": True}
        assert results[1] == {"also_valid": True}

    async def test_get_results_empty_responses(self, gemini_client):
        client, _ = gemini_client
        job = MagicMock()
        job.dest.inlined_responses = []

        results = await client.get_results(job)

        assert results == []

    async def test_get_results_no_dest(self, gemini_client):
        client, _ = gemini_client
        job = MagicMock(spec=[])  # No attributes at all

        results = await client.get_results(job)

        assert results == []


# ── Prompt Tests ──────────────────────────────────────────────────


class TestPrompts:
    def test_link_prompt_is_nonempty(self):
        assert isinstance(LINK_GENERATION_PROMPT, str)
        assert len(LINK_GENERATION_PROMPT) > 50

    def test_link_prompt_specifies_json_format(self):
        assert "source_id" in LINK_GENERATION_PROMPT
        assert "target_id" in LINK_GENERATION_PROMPT
        assert "link_type" in LINK_GENERATION_PROMPT
        assert "strength" in LINK_GENERATION_PROMPT

    def test_link_prompt_mentions_link_types(self):
        assert "semantic" in LINK_GENERATION_PROMPT
        assert "causal" in LINK_GENERATION_PROMPT
        assert "temporal" in LINK_GENERATION_PROMPT

    def test_pattern_prompt_is_nonempty(self):
        assert isinstance(PATTERN_DETECTION_PROMPT, str)
        assert len(PATTERN_DETECTION_PROMPT) > 50

    def test_pattern_prompt_specifies_json_format(self):
        assert "patterns" in PATTERN_DETECTION_PROMPT
        assert "content" in PATTERN_DETECTION_PROMPT
        assert "keywords" in PATTERN_DETECTION_PROMPT
        assert "supporting_note_ids" in PATTERN_DETECTION_PROMPT

    def test_contradiction_prompt_is_nonempty(self):
        assert isinstance(CONTRADICTION_DETECTION_PROMPT, str)
        assert len(CONTRADICTION_DETECTION_PROMPT) > 50

    def test_contradiction_prompt_specifies_json_format(self):
        assert "contradictions" in CONTRADICTION_DETECTION_PROMPT
        assert "note_id_a" in CONTRADICTION_DETECTION_PROMPT
        assert "note_id_b" in CONTRADICTION_DETECTION_PROMPT
        assert "description" in CONTRADICTION_DETECTION_PROMPT

    def test_profile_prompt_is_nonempty(self):
        assert isinstance(PROFILE_UPDATE_PROMPT, str)
        assert len(PROFILE_UPDATE_PROMPT) > 50

    def test_profile_prompt_references_all_sections(self):
        for section in PROFILE_SECTIONS:
            assert section in PROFILE_UPDATE_PROMPT

    def test_profile_prompt_mentions_max_facts(self):
        assert "30" in PROFILE_UPDATE_PROMPT

    def test_profile_prompt_excludes_expertise(self):
        prompt_lower = PROFILE_UPDATE_PROMPT.lower()
        assert "skill level" in prompt_lower or "expertise" in prompt_lower


# ── Task Builder Tests ────────────────────────────────────────────


def _assert_request_structure(req: dict, expected_temp: float) -> None:
    """Verify a request dict has the correct top-level structure."""
    assert "contents" in req
    assert "system_instruction" in req
    assert "generation_config" in req

    assert len(req["contents"]) == 1
    assert req["contents"][0]["role"] == "user"
    assert "text" in req["contents"][0]["parts"][0]

    assert "text" in req["system_instruction"]["parts"][0]

    assert req["generation_config"]["temperature"] == expected_temp
    assert req["generation_config"]["response_mime_type"] == "application/json"


class TestBuildLinkRequests:
    def test_structure(self):
        notes = [_make_note(f"n{i}") for i in range(3)]
        requests = build_link_requests(notes, [])

        assert len(requests) == 1
        _assert_request_structure(requests[0], DREAMER_LINK_TEMPERATURE)

    def test_batching(self):
        notes = [_make_note(f"n{i}") for i in range(25)]
        requests = build_link_requests(notes, [])

        assert len(requests) == 2

    def test_includes_existing_links(self):
        notes = [_make_note("n1"), _make_note("n2")]
        existing = [_make_link("n1", "n2")]
        requests = build_link_requests(notes, existing)

        user_text = requests[0]["contents"][0]["parts"][0]["text"]
        payload = json.loads(user_text)
        assert len(payload["existing_links_to_skip"]) > 0

    def test_empty_notes(self):
        requests = build_link_requests([], [])
        assert requests == []


class TestBuildPatternRequests:
    def test_structure(self):
        notes = [_make_note("n1", session_id="s1")]
        sessions = [_make_session("s1")]
        requests = build_pattern_requests(notes, sessions)

        assert len(requests) == 1
        _assert_request_structure(requests[0], DREAMER_PATTERN_TEMPERATURE)

    def test_groups_by_session(self):
        notes = [
            _make_note("n1", session_id="s1"),
            _make_note("n2", session_id="s1"),
            _make_note("n3", session_id="s2"),
        ]
        sessions = [_make_session("s1"), _make_session("s2")]
        requests = build_pattern_requests(notes, sessions)

        user_text = requests[0]["contents"][0]["parts"][0]["text"]
        payload = json.loads(user_text)
        assert len(payload["sessions"]) == 2

    def test_includes_session_metadata(self):
        notes = [_make_note("n1", session_id="s1")]
        sessions = [_make_session("s1")]
        requests = build_pattern_requests(notes, sessions)

        user_text = requests[0]["contents"][0]["parts"][0]["text"]
        payload = json.loads(user_text)
        assert payload["sessions"][0]["started_at"] == "2026-01-01T00:00:00.000Z"


class TestBuildContradictionRequests:
    def test_structure(self):
        pairs = [(_make_note("n1"), _make_note("n2"))]
        requests = build_contradiction_requests(pairs)

        assert len(requests) == 1
        _assert_request_structure(requests[0], DREAMER_CONTRADICTION_TEMPERATURE)

    def test_batching(self):
        pairs = [(_make_note(f"a{i}"), _make_note(f"b{i}")) for i in range(15)]
        requests = build_contradiction_requests(pairs)

        assert len(requests) == 2

    def test_pair_content_included(self):
        pairs = [(_make_note("n1", content="fact A"), _make_note("n2", content="fact B"))]
        requests = build_contradiction_requests(pairs)

        user_text = requests[0]["contents"][0]["parts"][0]["text"]
        payload = json.loads(user_text)
        assert payload["pairs"][0]["note_a"]["content"] == "fact A"
        assert payload["pairs"][0]["note_b"]["content"] == "fact B"

    def test_empty_pairs(self):
        requests = build_contradiction_requests([])
        assert requests == []


class TestBuildProfileRequest:
    def test_structure(self):
        notes = [_make_note("n1", importance=0.8)]
        req = build_profile_request(notes, None)

        assert isinstance(req, dict)
        _assert_request_structure(req, DREAMER_PROFILE_TEMPERATURE)

    def test_is_single_dict(self):
        notes = [_make_note("n1")]
        req = build_profile_request(notes, None)

        assert not isinstance(req, list)

    def test_includes_current_profile(self):
        notes = [_make_note("n1")]
        profile = {"identity": ["Name: Alice"], "professional": []}
        req = build_profile_request(notes, profile)

        user_text = req["contents"][0]["parts"][0]["text"]
        payload = json.loads(user_text)
        assert "current_profile" in payload
        assert payload["current_profile"] == profile

    def test_without_current_profile(self):
        notes = [_make_note("n1")]
        req = build_profile_request(notes, None)

        user_text = req["contents"][0]["parts"][0]["text"]
        payload = json.loads(user_text)
        assert "current_profile" not in payload

    def test_notes_sorted_by_importance(self):
        notes = [
            _make_note("n1", importance=0.3),
            _make_note("n2", importance=0.9),
            _make_note("n3", importance=0.6),
        ]
        req = build_profile_request(notes, None)

        user_text = req["contents"][0]["parts"][0]["text"]
        payload = json.loads(user_text)
        importances = [n["importance"] for n in payload["notes"]]
        assert importances == sorted(importances, reverse=True)
