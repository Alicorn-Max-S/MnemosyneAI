"""Tests for the MAGMA multi-graph module."""

import pytest

from mnemosyne.db.sqlite_store import SQLiteStore
from mnemosyne.graph.magma import MAGMAGraph


@pytest.fixture
async def magma(store: SQLiteStore) -> MAGMAGraph:
    """Create a MAGMAGraph backed by the test store."""
    return MAGMAGraph(store)


# ── Entity Extraction ──────────────────────────────────────────────


class TestExtractEntities:
    def test_extracts_capitalized_multi_word_names(self, magma):
        entities = magma.extract_entities("I met John Smith at the park")
        names = [e[0] for e in entities]
        assert "John Smith" in names

    def test_person_type(self, magma):
        entities = magma.extract_entities("I talked to John Smith yesterday")
        name_types = {e[0]: e[1] for e in entities}
        assert name_types.get("John Smith") == "person"

    def test_extracts_at_patterns(self, magma):
        entities = magma.extract_entities("Talked to @JaneDoe about the project")
        names = [e[0] for e in entities]
        assert "JaneDoe" in names

    def test_at_pattern_is_person(self, magma):
        entities = magma.extract_entities("Message from @Bob")
        name_types = {e[0]: e[1] for e in entities}
        assert name_types.get("Bob") == "person"

    def test_skips_sentence_starters(self, magma):
        entities = magma.extract_entities("The cat sat on the mat.")
        assert entities == []

    def test_empty_string(self, magma):
        assert magma.extract_entities("") == []

    def test_no_entities_in_lowercase(self, magma):
        assert magma.extract_entities("hello world foo bar") == []

    def test_deduplicates(self, magma):
        entities = magma.extract_entities(
            "I saw John Smith. Later I met John Smith again."
        )
        names = [e[0] for e in entities]
        assert names.count("John Smith") == 1

    def test_multiple_entities(self, magma):
        entities = magma.extract_entities(
            "I met John Smith and Jane Doe at the conference"
        )
        names = [e[0] for e in entities]
        assert "John Smith" in names
        assert "Jane Doe" in names


# ── Add Note Entities ──────────────────────────────────────────────


class TestAddNoteEntities:
    async def test_persists_to_sqlite(self, store, magma):
        peer = await store.create_peer("Alice")
        note = await store.create_note(peer.id, "Met Bob Smith at work")

        await magma.add_note_entities(note, [("Bob Smith", "person")])

        mentions = await store.get_entity_mentions(peer.id, "Bob Smith")
        assert len(mentions) >= 1
        assert mentions[0]["entity_name"] == "Bob Smith"
        assert mentions[0]["entity_type"] == "person"

    async def test_updates_in_memory_graph(self, store, magma):
        peer = await store.create_peer("Alice")
        note = await store.create_note(peer.id, "Bob and Carol meeting")

        await magma.add_note_entities(
            note, [("Bob Smith", "person"), ("Carol Lee", "person")]
        )

        assert magma._entity_graph.has_node("Bob Smith")
        assert magma._entity_graph.has_node("Carol Lee")
        assert magma._entity_graph.has_edge("Bob Smith", "Carol Lee")

    async def test_increments_edge_weight(self, store, magma):
        peer = await store.create_peer("Alice")
        note1 = await store.create_note(peer.id, "Bob and Carol first meeting")
        note2 = await store.create_note(peer.id, "Bob and Carol second meeting")

        await magma.add_note_entities(
            note1, [("Bob Smith", "person"), ("Carol Lee", "person")]
        )
        await magma.add_note_entities(
            note2, [("Bob Smith", "person"), ("Carol Lee", "person")]
        )

        weight = magma._entity_graph["Bob Smith"]["Carol Lee"]["weight"]
        assert weight == 2


# ── Load ───────────────────────────────────────────────────────────


class TestLoad:
    async def test_load_rebuilds_graph(self, store, magma):
        peer = await store.create_peer("Alice")
        note = await store.create_note(peer.id, "Meeting with Bob and Carol")

        await store.add_entity_mention(note.id, peer.id, "Bob", "person")
        await store.add_entity_mention(note.id, peer.id, "Carol", "person")

        await magma.load(peer.id)

        assert magma._entity_graph.has_node("Bob")
        assert magma._entity_graph.has_node("Carol")
        assert magma._entity_graph.has_edge("Bob", "Carol")

    async def test_load_clears_previous(self, store, magma):
        peer = await store.create_peer("Alice")
        magma._entity_graph.add_node("OldEntity")

        await magma.load(peer.id)

        assert "OldEntity" not in magma._entity_graph


# ── Get Related Entities ───────────────────────────────────────────


class TestGetRelatedEntities:
    async def test_returns_neighbors_sorted_by_weight(self, store, magma):
        peer = await store.create_peer("Alice")

        # Create notes with co-occurring entities
        note1 = await store.create_note(peer.id, "Meeting 1")
        note2 = await store.create_note(peer.id, "Meeting 2")
        note3 = await store.create_note(peer.id, "Meeting 3")

        # Bob co-occurs with Carol 2 times and Dave 1 time
        await magma.add_note_entities(
            note1, [("Bob", "person"), ("Carol", "person")]
        )
        await magma.add_note_entities(
            note2, [("Bob", "person"), ("Carol", "person")]
        )
        await magma.add_note_entities(
            note3, [("Bob", "person"), ("Dave", "person")]
        )

        magma._loaded_peer = peer.id
        related = magma.get_related_entities("Bob", peer.id)

        assert related[0] == "Carol"  # Weight 2
        assert "Dave" in related      # Weight 1

    def test_unknown_entity_returns_empty(self, magma):
        magma._loaded_peer = "peer_1"
        assert magma.get_related_entities("Nobody", "peer_1") == []

    def test_wrong_peer_returns_empty(self, magma):
        magma._loaded_peer = "peer_1"
        magma._entity_graph.add_node("Bob")
        assert magma.get_related_entities("Bob", "peer_2") == []

    def test_top_k_limit(self, store, magma):
        magma._loaded_peer = "p1"
        magma._entity_graph.add_node("Center")
        for i in range(5):
            name = f"Entity{i}"
            magma._entity_graph.add_node(name)
            magma._entity_graph.add_edge("Center", name, weight=i + 1)

        related = magma.get_related_entities("Center", "p1", top_k=3)
        assert len(related) == 3


# ── Get Entity Subgraph ───────────────────────────────────────────


class TestGetEntitySubgraph:
    def test_returns_adjacency_dict(self, magma):
        magma._entity_graph.add_edge("A", "B", weight=1)
        magma._entity_graph.add_edge("B", "C", weight=1)

        sub = magma.get_entity_subgraph("A", depth=2)

        assert "A" in sub
        assert "B" in sub
        assert "C" in sub

    def test_depth_1_limits_scope(self, magma):
        magma._entity_graph.add_edge("A", "B", weight=1)
        magma._entity_graph.add_edge("B", "C", weight=1)

        sub = magma.get_entity_subgraph("A", depth=1)

        assert "A" in sub
        assert "B" in sub
        assert "C" not in sub

    def test_unknown_entity_returns_empty(self, magma):
        assert magma.get_entity_subgraph("Nobody") == {}


# ── Get Communities ────────────────────────────────────────────────


class TestGetCommunities:
    def test_returns_groups(self, magma):
        magma._loaded_peer = "p1"

        # Cluster 1: A-B-C fully connected
        magma._entity_graph.add_edge("A", "B", weight=3)
        magma._entity_graph.add_edge("B", "C", weight=3)
        magma._entity_graph.add_edge("A", "C", weight=3)

        # Cluster 2: D-E-F fully connected
        magma._entity_graph.add_edge("D", "E", weight=3)
        magma._entity_graph.add_edge("E", "F", weight=3)
        magma._entity_graph.add_edge("D", "F", weight=3)

        communities = magma.get_communities("p1")

        assert len(communities) >= 2
        # Each community should have multiple members
        for comm in communities:
            assert len(comm) >= 2

    def test_empty_graph_returns_empty(self, magma):
        magma._loaded_peer = "p1"
        assert magma.get_communities("p1") == []

    def test_wrong_peer_returns_empty(self, magma):
        magma._loaded_peer = "p1"
        magma._entity_graph.add_edge("A", "B", weight=1)
        assert magma.get_communities("p2") == []
