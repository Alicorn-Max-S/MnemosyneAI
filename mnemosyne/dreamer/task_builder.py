"""Builds Gemini Batch API request payloads from post-dedup notes."""

import json
import logging

from mnemosyne.config import (
    DREAMER_CONTRADICTION_TEMPERATURE,
    DREAMER_LINK_TEMPERATURE,
    DREAMER_PATTERN_TEMPERATURE,
    DREAMER_PROFILE_TEMPERATURE,
)
from mnemosyne.dreamer.prompts import (
    CONTRADICTION_DETECTION_PROMPT,
    LINK_GENERATION_PROMPT,
    PATTERN_DETECTION_PROMPT,
    PROFILE_UPDATE_PROMPT,
)
from mnemosyne.models import Link, Note, Session

logger = logging.getLogger(__name__)

_LINK_BATCH_SIZE = 20
_CONTRADICTION_BATCH_SIZE = 10


def _build_request(user_text: str, system_prompt: str, temperature: float) -> dict:
    """Build a single Gemini batch request dict.

    Args:
        user_text: The user message content.
        system_prompt: The system instruction text.
        temperature: Generation temperature.

    Returns:
        A request dict in the format expected by GeminiClient.submit_batch.
    """
    return {
        "contents": [{"parts": [{"text": user_text}], "role": "user"}],
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "generation_config": {
            "temperature": temperature,
            "response_mime_type": "application/json",
        },
    }


def build_link_requests(notes: list[Note], existing_links: list[Link]) -> list[dict]:
    """Build batch requests for link generation.

    Groups notes into batches of ~20. Each request includes note IDs/content
    and existing links to avoid duplicates.

    Args:
        notes: Notes to analyze for potential links.
        existing_links: Already-created links to skip.

    Returns:
        List of request dicts for the batch API.
    """
    existing_pairs = {
        (link.source_note_id, link.target_note_id) for link in existing_links
    } | {
        (link.target_note_id, link.source_note_id) for link in existing_links
    }

    note_dicts = [
        {"id": n.id, "content": n.content, "keywords": n.keywords}
        for n in notes
    ]

    requests: list[dict] = []
    for i in range(0, len(note_dicts), _LINK_BATCH_SIZE):
        batch = note_dicts[i : i + _LINK_BATCH_SIZE]
        batch_ids = {nd["id"] for nd in batch}

        relevant_existing = [
            [a, b] for a, b in existing_pairs
            if a in batch_ids or b in batch_ids
        ]

        user_text = json.dumps({
            "notes": batch,
            "existing_links_to_skip": relevant_existing,
        })
        requests.append(_build_request(user_text, LINK_GENERATION_PROMPT, DREAMER_LINK_TEMPERATURE))

    logger.info("Built %d link generation requests from %d notes", len(requests), len(notes))
    return requests


def build_pattern_requests(notes: list[Note], sessions: list[Session]) -> list[dict]:
    """Build batch requests for cross-session pattern detection.

    Groups notes by session and includes session metadata for context.

    Args:
        notes: Notes to analyze for patterns.
        sessions: Session objects providing metadata.

    Returns:
        List of request dicts for the batch API.
    """
    session_map = {s.id: s for s in sessions}

    groups: dict[str | None, list[dict]] = {}
    for note in notes:
        sid = note.session_id
        if sid not in groups:
            groups[sid] = []
        groups[sid].append({"id": note.id, "content": note.content})

    session_data: list[dict] = []
    for sid, note_list in groups.items():
        session_info: dict = {"session_id": sid, "notes": note_list}
        if sid and sid in session_map:
            s = session_map[sid]
            session_info["started_at"] = s.started_at
            if s.summary:
                session_info["summary"] = s.summary
        session_data.append(session_info)

    user_text = json.dumps({"sessions": session_data})
    requests = [_build_request(user_text, PATTERN_DETECTION_PROMPT, DREAMER_PATTERN_TEMPERATURE)]

    logger.info("Built %d pattern detection requests from %d notes across %d sessions",
                len(requests), len(notes), len(session_data))
    return requests


def build_contradiction_requests(
    candidate_pairs: list[tuple[Note, Note]],
) -> list[dict]:
    """Build batch requests for contradiction detection.

    Groups note pairs into batches of ~10 for evaluation.

    Args:
        candidate_pairs: Pairs of notes with high cosine similarity to evaluate.

    Returns:
        List of request dicts for the batch API.
    """
    requests: list[dict] = []
    for i in range(0, len(candidate_pairs), _CONTRADICTION_BATCH_SIZE):
        batch = candidate_pairs[i : i + _CONTRADICTION_BATCH_SIZE]
        pairs_data = [
            {
                "note_a": {"id": a.id, "content": a.content},
                "note_b": {"id": b.id, "content": b.content},
            }
            for a, b in batch
        ]
        user_text = json.dumps({"pairs": pairs_data})
        requests.append(_build_request(
            user_text, CONTRADICTION_DETECTION_PROMPT, DREAMER_CONTRADICTION_TEMPERATURE,
        ))

    logger.info("Built %d contradiction requests from %d pairs", len(requests), len(candidate_pairs))
    return requests


def build_profile_request(
    permanent_notes: list[Note],
    current_profile: dict | None,
) -> dict:
    """Build a single batch request for profile generation/update.

    Args:
        permanent_notes: All permanent notes for the peer, used as input.
        current_profile: The current profile dict, or None if no profile exists.

    Returns:
        A single request dict for the batch API.
    """
    sorted_notes = sorted(permanent_notes, key=lambda n: n.importance, reverse=True)
    note_data = [
        {"id": n.id, "content": n.content, "importance": n.importance}
        for n in sorted_notes
    ]

    payload: dict = {"notes": note_data}
    if current_profile is not None:
        payload["current_profile"] = current_profile

    user_text = json.dumps(payload)
    logger.info("Built profile request from %d permanent notes", len(permanent_notes))
    return _build_request(user_text, PROFILE_UPDATE_PROMPT, DREAMER_PROFILE_TEMPERATURE)
