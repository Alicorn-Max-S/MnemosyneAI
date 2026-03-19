"""Deriver: LLM-powered atomic fact extraction and scoring via NousResearch API."""

import asyncio
import json
import logging

import httpx

from mnemosyne.config import (
    DERIVER_EXTRACT_TEMPERATURE,
    DERIVER_MAX_RETRIES,
    DERIVER_RETRY_DELAYS,
    DERIVER_SCORE_TEMPERATURE,
    NOUSRESEARCH_BASE_URL,
    NOUSRESEARCH_MODEL,
)

logger = logging.getLogger(__name__)


class DeriverError(Exception):
    """Base exception for Deriver failures."""


class DeriverAPIError(DeriverError):
    """Raised on unrecoverable API errors (non-retryable status codes, exhausted retries)."""


class DeriverParseError(DeriverError):
    """Raised when the API returns unparseable JSON too many times."""


EXTRACT_SYSTEM_PROMPT = """\
You extract atomic facts from a user message in a conversation.

Rules:
- Extract facts ONLY from the <user_message> block. Never extract from assistant messages.
- Each fact must be a single, self-contained statement.
- If the user message confirms or agrees with something the assistant said, mark that \
note with "is_confirmation": true and state the confirmed fact explicitly.
- If the user message contains no extractable facts, return an empty list.

Return JSON: {"notes": [{"text": "...", "is_confirmation": false}, ...]}
"""

SCORE_SYSTEM_PROMPT = """\
You score and tag a list of atomic fact notes extracted from a conversation.

For each note, add these fields:
- emotional_weight: float 0.0-1.0, how emotionally significant this fact is
- provenance: one of "organic", "agent_prompted", "user_confirmed", "inferred". \
If the input note has "is_confirmation": true, set provenance to "user_confirmed".
- durability: one of "permanent", "contextual", "ephemeral"
- keywords: list of 1-5 keyword strings
- tags: list of 0-3 category tags
- context_description: one sentence describing when/why this fact is relevant

Preserve the original "text" field. Drop "is_confirmation" from output.

Return JSON: {"scored_notes": [{"text": "...", "emotional_weight": 0.5, \
"provenance": "organic", "durability": "permanent", "keywords": ["..."], \
"tags": ["..."], "context_description": "..."}, ...]}
"""


class Deriver:
    """Calls NousResearch API to extract and score atomic facts from messages."""

    def __init__(
        self,
        api_key: str,
        base_url: str = NOUSRESEARCH_BASE_URL,
        model: str = NOUSRESEARCH_MODEL,
    ) -> None:
        """Initialize with API credentials and optional endpoint/model overrides."""
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    async def _call_api(self, messages: list[dict], temperature: float) -> dict:
        """Send a chat completion request with retry logic. Returns parsed JSON content."""
        body = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        parse_failures = 0

        for attempt in range(DERIVER_MAX_RETRIES):
            try:
                response = await self._client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers,
                    json=body,
                )
            except httpx.HTTPError as exc:
                logger.warning("HTTP error on attempt %d: %s", attempt + 1, exc)
                if attempt < DERIVER_MAX_RETRIES - 1:
                    await asyncio.sleep(DERIVER_RETRY_DELAYS[attempt])
                    continue
                raise DeriverAPIError(f"All {DERIVER_MAX_RETRIES} attempts failed") from exc

            status = response.status_code
            if status == 429 or status >= 500:
                logger.warning("Retryable status %d on attempt %d", status, attempt + 1)
                if attempt < DERIVER_MAX_RETRIES - 1:
                    await asyncio.sleep(DERIVER_RETRY_DELAYS[attempt])
                    continue
                raise DeriverAPIError(f"API returned {status} after {DERIVER_MAX_RETRIES} attempts")

            if 400 <= status < 500:
                raise DeriverAPIError(f"API returned non-retryable status {status}")

            # 2xx — parse the response
            try:
                content_str = response.json()["choices"][0]["message"]["content"]
                return json.loads(content_str)
            except (json.JSONDecodeError, KeyError, IndexError) as exc:
                parse_failures += 1
                logger.warning("Parse failure %d on attempt %d: %s", parse_failures, attempt + 1, exc)
                if parse_failures >= 2:
                    raise DeriverParseError(
                        f"Failed to parse API response {parse_failures} times"
                    ) from exc
                if attempt < DERIVER_MAX_RETRIES - 1:
                    await asyncio.sleep(DERIVER_RETRY_DELAYS[attempt])
                    continue
                raise DeriverParseError("Parse failed on final attempt") from exc

        raise DeriverAPIError(f"All {DERIVER_MAX_RETRIES} attempts failed")

    async def extract(self, user_message: str, preceding_turns: list[dict]) -> list[dict]:
        """Extract atomic fact notes from a user message.

        Args:
            user_message: The user message to extract facts from.
            preceding_turns: Recent conversation turns as [{"role": ..., "content": ...}].

        Returns:
            List of note dicts with 'text' and 'is_confirmation' fields, or [] on failure.
        """
        try:
            parts: list[str] = []
            if preceding_turns:
                conversation_lines = "\n".join(
                    f"{t['role']}: {t['content']}" for t in preceding_turns
                )
                parts.append(f"<preceding_conversation>\n{conversation_lines}\n</preceding_conversation>")
            parts.append(f"<user_message>\n{user_message}\n</user_message>")
            user_content = "\n\n".join(parts)

            messages = [
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            result = await self._call_api(messages, DERIVER_EXTRACT_TEMPERATURE)
            return result.get("notes", [])
        except Exception:
            logger.exception("Extract failed, returning empty list")
            return []

    async def score(self, notes: list[dict]) -> list[dict]:
        """Score and tag a list of atomic fact notes.

        Args:
            notes: List of note dicts from extract().

        Returns:
            List of scored note dicts with metadata fields, or [] on failure.
        """
        if not notes:
            return []

        try:
            messages = [
                {"role": "system", "content": SCORE_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(notes)},
            ]
            result = await self._call_api(messages, DERIVER_SCORE_TEMPERATURE)
            return result.get("scored_notes", [])
        except Exception:
            logger.exception("Score failed, returning empty list")
            return []

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
