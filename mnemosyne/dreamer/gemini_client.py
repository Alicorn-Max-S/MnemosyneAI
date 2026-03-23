"""Gemini Batch API wrapper for Dreamer background processing."""

import asyncio
import json
import logging
import time

from google import genai

from mnemosyne.config import (
    DREAMER_MAX_POLL_TIME,
    DREAMER_POLL_INTERVAL,
    GEMINI_MODEL,
)

logger = logging.getLogger(__name__)


class GeminiError(Exception):
    """Base exception for Gemini Batch API failures."""


class GeminiTimeoutError(GeminiError):
    """Raised when polling exceeds the maximum allowed time."""


class GeminiBatchError(GeminiError):
    """Raised when a batch job ends in FAILED, CANCELLED, or EXPIRED state."""


_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

_FAILURE_STATES = {
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


class GeminiClient:
    """Thin async wrapper around the google-genai SDK for batch operations."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the Gemini client.

        Args:
            api_key: Gemini API key. If None, the SDK reads from GEMINI_API_KEY env var.
        """
        if api_key is not None:
            self._client = genai.Client(api_key=api_key)
        else:
            self._client = genai.Client()
        logger.info("GeminiClient initialized")

    async def submit_batch(self, requests: list[dict], display_name: str) -> str:
        """Submit a batch of requests to the Gemini Batch API.

        Args:
            requests: List of request dicts in Gemini batch format.
            display_name: Human-readable name for the batch job.

        Returns:
            The batch job name string.
        """
        job = await asyncio.to_thread(
            self._client.batches.create,
            model=GEMINI_MODEL,
            src=requests,
            config={"display_name": display_name},
        )
        logger.info("Submitted batch '%s' with %d requests: %s", display_name, len(requests), job.name)
        return job.name

    async def poll_until_done(
        self,
        job_name: str,
        poll_interval: float = DREAMER_POLL_INTERVAL,
        max_time: float = DREAMER_MAX_POLL_TIME,
    ) -> object:
        """Poll a batch job until it reaches a terminal state.

        Args:
            job_name: The batch job name returned by submit_batch.
            poll_interval: Seconds between poll attempts.
            max_time: Maximum total seconds to wait before raising timeout.

        Returns:
            The completed batch job object.

        Raises:
            GeminiTimeoutError: If max_time is exceeded.
            GeminiBatchError: If the job ends in a failure state.
        """
        start = time.monotonic()
        state = None

        while True:
            job = await asyncio.to_thread(self._client.batches.get, name=job_name)
            state = job.state.name if hasattr(job.state, "name") else str(job.state)
            logger.debug("Batch %s state: %s", job_name, state)

            if state in _TERMINAL_STATES:
                break

            elapsed = time.monotonic() - start
            if elapsed >= max_time:
                raise GeminiTimeoutError(
                    f"Batch {job_name} still in state {state} after {elapsed:.0f}s"
                )

            await asyncio.sleep(poll_interval)

        if state in _FAILURE_STATES:
            raise GeminiBatchError(f"Batch {job_name} ended in state: {state}")

        logger.info("Batch %s completed successfully", job_name)
        return job

    async def get_results(self, job: object) -> list[dict]:
        """Extract and parse results from a completed batch job.

        Args:
            job: A completed batch job object from poll_until_done.

        Returns:
            List of parsed JSON dicts from each response.
        """
        results: list[dict] = []
        responses = getattr(getattr(job, "dest", None), "inlined_responses", None)

        if not responses:
            logger.warning("No inlined responses found in batch job")
            return results

        for idx, response in enumerate(responses):
            try:
                text = response.candidates[0].content.parts[0].text
                parsed = json.loads(text)
                results.append(parsed)
            except (json.JSONDecodeError, AttributeError, IndexError, TypeError) as exc:
                logger.warning("Failed to parse response %d: %s", idx, exc)
                continue

        logger.info("Parsed %d/%d batch responses", len(results), len(responses))
        return results
