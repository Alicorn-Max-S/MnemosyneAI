"""MAGMA multi-graph: entity extraction, co-occurrence graph, community detection."""

import logging
import re
from collections import defaultdict
from itertools import combinations

import networkx as nx

from mnemosyne.db.sqlite_store import SQLiteStore
from mnemosyne.models import Note

logger = logging.getLogger(__name__)

# Words that are commonly capitalized but are not entities.
_STOPWORDS = frozenset({
    "The", "This", "That", "They", "There", "These", "Those",
    "What", "When", "Where", "Which", "Who", "Whom", "Whose",
    "How", "And", "But", "Or", "Nor", "For", "Yet", "So",
    "It", "He", "She", "We", "My", "Your", "His", "Her",
    "Its", "Our", "Their", "Is", "Are", "Was", "Were",
    "Has", "Have", "Had", "Do", "Does", "Did",
    "Can", "Could", "Would", "Should", "Will", "May", "Might",
    "Not", "No", "Yes", "Also", "Just", "Very", "Too",
    "If", "Then", "Than", "Some", "Any", "All", "Each", "Every",
    "Much", "Many", "Most", "Other", "Another", "Such",
    "Here", "Now", "Well", "Still", "Already", "Even",
    "However", "Although", "Because", "Since", "While",
    "After", "Before", "About", "Into", "Over", "With",
    "From", "Between", "Through", "During", "Without",
    "Again", "Further", "Once", "Both", "Few", "More",
    "Been", "Being", "Having", "Going", "Getting",
    "Today", "Tomorrow", "Yesterday",
    "I", "Me", "You", "Us", "Them",
})

# Pattern for @-mentions like @JohnDoe
_AT_PATTERN = re.compile(r"@(\w+)")

# Pattern for capitalized word sequences (2-4 words)
_CAP_SEQUENCE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")

# Simple sentence splitter
_SENTENCE_SPLIT = re.compile(r"[.!?]+\s+|[\n\r]+")


class MAGMAGraph:
    """Entity co-occurrence graph built from note entity mentions."""

    def __init__(self, db: SQLiteStore) -> None:
        """Initialize with database reference and empty graphs."""
        self._db = db
        self._entity_graph: nx.Graph = nx.Graph()
        self._loaded_peer: str | None = None

    async def load(self, peer_id: str) -> None:
        """Load entity graph from SQLite for a peer.

        Clears the current graph and rebuilds from entity_mentions table.
        """
        self._entity_graph.clear()
        self._loaded_peer = peer_id

        entities = await self._db.get_entities_for_peer(peer_id)
        if not entities:
            logger.debug("No entities found for peer %s", peer_id)
            return

        # Build note_id → [entity_name] mapping
        note_entities: dict[str, list[str]] = defaultdict(list)
        for entity_info in entities:
            name = entity_info["entity_name"]
            entity_type = entity_info.get("entity_type", "other")
            self._entity_graph.add_node(name, entity_type=entity_type)

            mentions = await self._db.get_entity_mentions(peer_id, name)
            for mention in mentions:
                note_entities[mention["note_id"]].append(name)

        # Add edges for co-occurring entities in the same note
        for _note_id, ent_names in note_entities.items():
            unique_names = list(set(ent_names))
            for a, b in combinations(unique_names, 2):
                if self._entity_graph.has_edge(a, b):
                    self._entity_graph[a][b]["weight"] += 1
                else:
                    self._entity_graph.add_edge(a, b, weight=1)

        logger.info(
            "Loaded entity graph for peer %s: %d nodes, %d edges",
            peer_id,
            self._entity_graph.number_of_nodes(),
            self._entity_graph.number_of_edges(),
        )

    def extract_entities(self, text: str) -> list[tuple[str, str]]:
        """Extract entities from text using rule-based heuristics.

        Returns list of (entity_name, entity_type) tuples. Best-effort:
        returns empty list on any failure.
        """
        try:
            if not text or not text.strip():
                return []

            entities: list[tuple[str, str]] = []
            seen: set[str] = set()

            # Extract @-mentions
            for match in _AT_PATTERN.finditer(text):
                name = match.group(1)
                if name not in seen:
                    seen.add(name)
                    entities.append((name, "person"))

            # Split into sentences to detect sentence-start words
            sentences = _SENTENCE_SPLIT.split(text)

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Find capitalized multi-word sequences
                for match in _CAP_SEQUENCE.finditer(sentence):
                    name = match.group(1)
                    # Skip if it starts at position 0 (sentence start)
                    if match.start() == 0:
                        continue
                    # Skip if any word is a stopword
                    words = name.split()
                    if any(w in _STOPWORDS for w in words):
                        continue
                    if name not in seen:
                        seen.add(name)
                        entities.append((name, "person"))

            return entities

        except Exception:
            logger.exception("Entity extraction failed")
            return []

    async def add_note_entities(
        self, note: Note, entities: list[tuple[str, str]]
    ) -> None:
        """Persist entity mentions to SQLite and update in-memory graph.

        Best-effort: logs failures per entity, never raises.
        """
        added_names: list[str] = []

        for entity_name, entity_type in entities:
            try:
                context = note.content[:200] if note.content else None
                await self._db.add_entity_mention(
                    note_id=note.id,
                    peer_id=note.peer_id,
                    entity_name=entity_name,
                    entity_type=entity_type,
                    mention_context=context,
                )
                # Update in-memory graph
                if not self._entity_graph.has_node(entity_name):
                    self._entity_graph.add_node(
                        entity_name, entity_type=entity_type
                    )
                added_names.append(entity_name)
            except Exception:
                logger.warning(
                    "Failed to add entity mention %s for note %s",
                    entity_name,
                    note.id,
                    exc_info=True,
                )

        # Add/increment edges for all pairs of entities in this note
        for a, b in combinations(added_names, 2):
            if self._entity_graph.has_edge(a, b):
                self._entity_graph[a][b]["weight"] += 1
            else:
                self._entity_graph.add_edge(a, b, weight=1)

    def get_related_entities(
        self, entity_name: str, peer_id: str, top_k: int = 10
    ) -> list[str]:
        """Return entities most connected to the given entity by edge weight.

        Returns empty list if entity not in graph or wrong peer loaded.
        """
        if self._loaded_peer is not None and self._loaded_peer != peer_id:
            return []
        if entity_name not in self._entity_graph:
            return []

        neighbors = []
        for neighbor in self._entity_graph.neighbors(entity_name):
            weight = self._entity_graph[entity_name][neighbor].get("weight", 1)
            neighbors.append((neighbor, weight))

        neighbors.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _w in neighbors[:top_k]]

    def get_entity_subgraph(self, entity_name: str, depth: int = 2) -> dict:
        """Return BFS subgraph from entity as adjacency dict.

        Returns empty dict if entity not in graph.
        """
        if entity_name not in self._entity_graph:
            return {}

        try:
            sub = nx.ego_graph(self._entity_graph, entity_name, radius=depth)
            return nx.to_dict_of_lists(sub)
        except Exception:
            logger.exception(
                "Failed to get subgraph for entity %s", entity_name
            )
            return {}

    def get_communities(self, peer_id: str) -> list[list[str]]:
        """Detect communities in the entity graph.

        Returns groups of related entity names. Empty graph returns [].
        """
        if self._loaded_peer is not None and self._loaded_peer != peer_id:
            return []
        if (
            self._entity_graph.number_of_nodes() == 0
            or self._entity_graph.number_of_edges() == 0
        ):
            return []

        try:
            communities = nx.community.greedy_modularity_communities(
                self._entity_graph
            )
        except Exception:
            logger.warning(
                "Louvain failed, falling back to label propagation",
                exc_info=True,
            )
            try:
                communities = nx.community.label_propagation_communities(
                    self._entity_graph
                )
            except Exception:
                logger.exception("Community detection failed entirely")
                return []

        # Filter single-node communities and sort for determinism
        result = []
        for comm in communities:
            members = sorted(comm)
            if len(members) >= 2:
                result.append(members)

        return result
