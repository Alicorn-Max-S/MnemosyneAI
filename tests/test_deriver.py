"""Tests for mnemosyne.pipeline.deriver."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from mnemosyne.pipeline.deriver import Deriver


def make_chat_response(content_dict: dict, status_code: int = 200) -> httpx.Response:
    """Build a mock httpx.Response matching the chat completions shape."""
    body = {"choices": [{"message": {"content": json.dumps(content_dict)}}]}
    return httpx.Response(status_code, json=body)


def make_error_response(status_code: int) -> httpx.Response:
    """Build a mock httpx.Response with the given error status code."""
    return httpx.Response(status_code, json={"error": "mock error"})


@pytest.fixture
def deriver() -> Deriver:
    return Deriver(api_key="test-key", base_url="https://test.example.com/v1")


@pytest.mark.asyncio
async def test_extracts_atomic_notes(deriver: Deriver) -> None:
    """Mock 2-note response, verify list has 2 entries with correct text/is_confirmation."""
    notes_payload = {
        "notes": [
            {"text": "User likes coffee", "is_confirmation": False},
            {"text": "User lives in Berlin", "is_confirmation": False},
        ]
    }
    deriver._client.post = AsyncMock(return_value=make_chat_response(notes_payload))

    result = await deriver.extract("I like coffee and I live in Berlin", [])

    assert len(result) == 2
    assert result[0]["text"] == "User likes coffee"
    assert result[0]["is_confirmation"] is False
    assert result[1]["text"] == "User lives in Berlin"


@pytest.mark.asyncio
async def test_confirmation_detection(deriver: Deriver) -> None:
    """'yeah that's right' after assistant statement -> is_confirmation=True."""
    notes_payload = {
        "notes": [
            {"text": "User confirms they work at Acme Corp", "is_confirmation": True},
        ]
    }
    deriver._client.post = AsyncMock(return_value=make_chat_response(notes_payload))

    preceding = [
        {"role": "assistant", "content": "So you work at Acme Corp?"},
    ]
    result = await deriver.extract("yeah that's right", preceding)

    assert len(result) == 1
    assert result[0]["is_confirmation"] is True


@pytest.mark.asyncio
async def test_no_facts_returns_empty(deriver: Deriver) -> None:
    """Mock {"notes": []} -> returns []."""
    deriver._client.post = AsyncMock(return_value=make_chat_response({"notes": []}))

    result = await deriver.extract("hello", [])

    assert result == []


@pytest.mark.asyncio
async def test_scorer_returns_all_fields(deriver: Deriver) -> None:
    """Verify scored note has all 6 metadata fields with correct types."""
    scored_payload = {
        "scored_notes": [
            {
                "text": "User likes coffee",
                "emotional_weight": 0.3,
                "provenance": "organic",
                "durability": "permanent",
                "keywords": ["coffee", "preference"],
                "tags": ["food"],
                "context_description": "User's beverage preference",
            }
        ]
    }
    deriver._client.post = AsyncMock(return_value=make_chat_response(scored_payload))

    notes = [{"text": "User likes coffee", "is_confirmation": False}]
    result = await deriver.score(notes)

    assert len(result) == 1
    note = result[0]
    assert isinstance(note["emotional_weight"], float)
    assert note["provenance"] == "organic"
    assert note["durability"] == "permanent"
    assert isinstance(note["keywords"], list)
    assert isinstance(note["tags"], list)
    assert isinstance(note["context_description"], str)


@pytest.mark.asyncio
async def test_scorer_user_confirmed_passthrough(deriver: Deriver) -> None:
    """Note with is_confirmation=True -> provenance='user_confirmed'."""
    scored_payload = {
        "scored_notes": [
            {
                "text": "User works at Acme Corp",
                "emotional_weight": 0.2,
                "provenance": "user_confirmed",
                "durability": "permanent",
                "keywords": ["work", "acme"],
                "tags": ["employment"],
                "context_description": "User's workplace",
            }
        ]
    }
    deriver._client.post = AsyncMock(return_value=make_chat_response(scored_payload))

    notes = [{"text": "User works at Acme Corp", "is_confirmation": True}]
    result = await deriver.score(notes)

    assert len(result) == 1
    assert result[0]["provenance"] == "user_confirmed"


@pytest.mark.asyncio
async def test_retry_on_429(deriver: Deriver) -> None:
    """side_effect=[429_response, 200_response] -> returns results, called twice."""
    ok_payload = {"notes": [{"text": "fact", "is_confirmation": False}]}
    deriver._client.post = AsyncMock(
        side_effect=[
            make_error_response(429),
            make_chat_response(ok_payload),
        ]
    )

    with patch("mnemosyne.pipeline.deriver.asyncio.sleep", new_callable=AsyncMock):
        result = await deriver.extract("some message", [])

    assert len(result) == 1
    assert result[0]["text"] == "fact"
    assert deriver._client.post.call_count == 2


@pytest.mark.asyncio
async def test_garbage_json_returns_empty(deriver: Deriver) -> None:
    """Invalid JSON content -> returns [], no crash."""
    garbage_body = {"choices": [{"message": {"content": "not valid json {{"}}]}
    garbage_response = httpx.Response(200, json=garbage_body)
    deriver._client.post = AsyncMock(return_value=garbage_response)

    with patch("mnemosyne.pipeline.deriver.asyncio.sleep", new_callable=AsyncMock):
        result = await deriver.extract("some message", [])

    assert result == []


@pytest.mark.asyncio
async def test_empty_notes_no_api_call(deriver: Deriver) -> None:
    """score([]) -> [], post never called."""
    deriver._client.post = AsyncMock()

    result = await deriver.score([])

    assert result == []
    deriver._client.post.assert_not_called()
