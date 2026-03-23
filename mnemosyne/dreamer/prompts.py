"""System prompts for Dreamer batch tasks."""

LINK_GENERATION_PROMPT = """\
You analyze a set of memory notes and identify meaningful relationships between them.

For each pair of notes that share a meaningful relationship, create a link with:
- source_id: the ID of the first note
- target_id: the ID of the second note
- link_type: one of "semantic", "causal", "temporal", "contradicts", "supports", "derived_from"
- strength: float 0.0-1.0 indicating relationship strength

Link type guidelines:
- semantic: notes share a common topic or concept
- causal: one note describes a cause or effect of the other
- temporal: notes describe events in a time sequence
- contradicts: notes contain conflicting information
- supports: one note provides evidence or confirmation for the other
- derived_from: one note is a refinement or elaboration of the other

Rules:
- Only create links where there is a clear, meaningful relationship.
- Do not link every pair — most pairs will have no link.
- Existing links are provided; do not duplicate them.
- A pair of notes should have at most one link.

Return JSON only: {"links": [{"source_id": "...", "target_id": "...", "link_type": "...", \
"strength": 0.8}]}
If no links are found, return: {"links": []}
"""

PATTERN_DETECTION_PROMPT = """\
You analyze memory notes across multiple conversation sessions to detect cross-session \
patterns, recurring themes, and behavioral trends.

Each pattern you identify should be:
- A higher-order observation that synthesizes information across multiple notes
- Not a restatement of any single note
- Supported by at least 2 notes from different sessions when possible

For each pattern, provide:
- content: a single clear statement describing the pattern or trend
- keywords: 1-5 keyword strings for categorization
- supporting_note_ids: IDs of the notes that support this pattern

Rules:
- Focus on recurring behaviors, evolving preferences, and consistent themes.
- Do not fabricate patterns — every pattern must be grounded in the provided notes.
- Prefer quality over quantity. A few strong patterns are better than many weak ones.

Return JSON only: {"patterns": [{"content": "...", "keywords": ["..."], \
"supporting_note_ids": ["..."]}]}
If no patterns are found, return: {"patterns": []}
"""

CONTRADICTION_DETECTION_PROMPT = """\
You examine pairs of similar memory notes and determine whether they truly contradict \
each other.

A contradiction exists when two notes make incompatible claims about the same subject. \
These are NOT contradictions:
- A note that updates or refines an earlier note (evolution, not contradiction)
- Two notes about different time periods (preferences can change)
- Notes that are similar but not in conflict

For each genuine contradiction, provide:
- note_id_a: ID of the first note
- note_id_b: ID of the second note
- description: brief explanation of what specifically contradicts

Rules:
- Be conservative — only flag clear, unambiguous contradictions.
- Temporal context matters: "I like coffee" and "I stopped drinking coffee" is an update, \
not a contradiction, if the second is more recent.
- If unsure, do not flag it.

Return JSON only: {"contradictions": [{"note_id_a": "...", "note_id_b": "...", \
"description": "..."}]}
If no contradictions are found, return: {"contradictions": []}
"""

PROFILE_UPDATE_PROMPT = """\
You generate a concise static profile from a collection of permanent facts about a person.

Organize facts into exactly these 4 sections:
- identity: name, age, location, key personal identifiers
- professional: role, company, industry
- communication_style: preferred tone, formality, response length preferences
- relationships: important people, pets, family mentions

Rules:
- Maximum 30 facts total across all sections.
- Each fact is one short, self-contained sentence.
- Use empty lists for sections with no relevant facts.
- Prioritize facts with higher importance scores.
- Do NOT include skill levels, expertise, or technical proficiency — those belong in \
dynamic memory, not the static profile.
- Do NOT infer or hallucinate — only include facts directly supported by the input notes.
- If a current profile is provided, update it rather than starting from scratch. \
Remove facts no longer supported, add new ones.

Return JSON only: {"profile": {"identity": ["..."], "professional": ["..."], \
"communication_style": ["..."], "relationships": ["..."]}}
"""
