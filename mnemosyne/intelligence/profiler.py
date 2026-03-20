"""Static profile generator: builds Peer Cards from permanent notes via Deriver API."""

import logging

from mnemosyne.config import (
    DERIVER_EXTRACT_TEMPERATURE,
    PROFILE_MAX_FACTS,
    PROFILE_MIN_NOTES,
    PROFILE_SECTIONS,
)
from mnemosyne.db.sqlite_store import SQLiteStore
from mnemosyne.models import PeerProfile
from mnemosyne.pipeline.deriver import Deriver

logger = logging.getLogger(__name__)

PROFILE_SYSTEM_PROMPT = """\
You generate a concise static profile (Peer Card) from a collection of permanent facts \
about a person.

Organize the facts into exactly these sections:
- identity: name, age, location, key personal identifiers
- professional: role, company, industry (do NOT include skill level or expertise — \
those belong in dynamic memory)
- communication_style: preferred tone, formality, response length preferences
- relationships: important people, pets, family mentions

Rules:
- Maximum 30 facts total across all sections.
- Each fact is one short sentence.
- Omit sections with no relevant facts (return empty string for that section).
- Prioritize facts with higher importance scores.
- Do NOT infer or hallucinate — only include facts directly supported by the input notes.

Return JSON: {"identity": "...", "professional": "...", "communication_style": "...", \
"relationships": "..."}
"""

SECTION_TITLES = {
    "identity": "Identity",
    "professional": "Professional",
    "communication_style": "Communication Style",
    "relationships": "Relationships",
}


class Profiler:
    """Generates static Peer Card profiles from permanent notes via the Deriver API."""

    def __init__(self, db: SQLiteStore, deriver: Deriver) -> None:
        """Initialize with database and deriver references."""
        self._db = db
        self._deriver = deriver

    async def generate(self, peer_id: str) -> PeerProfile | None:
        """Generate a static profile for a peer from their permanent notes.

        Returns None if fewer than PROFILE_MIN_NOTES permanent notes exist
        or if the API call fails.
        """
        try:
            notes = await self._db.get_permanent_notes(peer_id, limit=50)
            if len(notes) < PROFILE_MIN_NOTES:
                logger.info(
                    "Peer %s has %d permanent notes, need %d — skipping profile",
                    peer_id, len(notes), PROFILE_MIN_NOTES,
                )
                return None

            note_lines = []
            for note in notes:
                note_lines.append(
                    f"- [importance={note.importance:.2f}] {note.content}"
                )
            user_content = "\n".join(note_lines)

            messages = [
                {"role": "system", "content": PROFILE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            result = await self._deriver._call_api(messages, DERIVER_EXTRACT_TEMPERATURE)

            sections: dict[str, str] = {}
            for section in PROFILE_SECTIONS:
                sections[section] = result.get(section, "")

            fact_count = sum(
                len([line for line in text.split("\n") if line.strip()])
                for text in sections.values()
                if text
            )
            fact_count = min(fact_count, PROFILE_MAX_FACTS)

            source_note_ids = [n.id for n in notes]

            profile = await self._db.upsert_profile(
                peer_id=peer_id,
                sections=sections,
                fact_count=fact_count,
                source_note_ids=source_note_ids,
            )
            logger.info("Generated profile for peer %s with %d facts", peer_id, fact_count)
            return profile

        except Exception:
            logger.warning("Profile generation failed for peer %s", peer_id, exc_info=True)
            return None

    async def get_profile_text(self, peer_id: str) -> str | None:
        """Fetch and format the profile as markdown text.

        Returns None if no profile exists.
        """
        profile = await self._db.get_profile(peer_id)
        if profile is None:
            return None

        peer = await self._db.get_peer(peer_id)
        peer_name = peer.name if peer else peer_id

        lines = [f"## About {peer_name}"]
        for section in PROFILE_SECTIONS:
            text = profile.sections.get(section, "")
            if text:
                title = SECTION_TITLES.get(section, section)
                lines.append(f"### {title}")
                lines.append(text)

        return "\n".join(lines)
