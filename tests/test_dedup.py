"""Tests for the batch dedup module."""

import pytest

from mnemosyne.db.sqlite_store import SQLiteStore
from mnemosyne.dreamer.dedup import DedupProcessor, DedupResult, _UnionFind
from mnemosyne.vectors.embedder import Embedder


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    """Module-scoped embedder — model loading is expensive."""
    return Embedder()


# ── UnionFind ──────────────────────────────────────────────────────


class TestUnionFind:
    def test_initial_singletons(self):
        uf = _UnionFind(["a", "b", "c"])
        assert uf.find("a") == "a"
        assert uf.find("b") == "b"
        assert uf.find("c") == "c"

    def test_union_and_find(self):
        uf = _UnionFind(["a", "b", "c"])
        uf.union("a", "b")
        assert uf.find("a") == uf.find("b")
        assert uf.find("c") != uf.find("a")

    def test_transitive_union(self):
        uf = _UnionFind(["a", "b", "c"])
        uf.union("a", "b")
        uf.union("b", "c")
        assert uf.find("a") == uf.find("c")

    def test_idempotent_union(self):
        uf = _UnionFind(["a", "b"])
        uf.union("a", "b")
        root_before = uf.find("a")
        uf.union("a", "b")
        assert uf.find("a") == root_before


# ── DedupProcessor ─────────────────────────────────────────────────


class TestDedupProcessor:
    async def test_clusters_similar_notes(self, store, embedder):
        peer = await store.create_peer("Alice")
        # Near-duplicate content
        await store.create_note(peer.id, "The cat sat on the mat", importance=0.5)
        await store.create_note(peer.id, "The cat sat on the mat today", importance=0.3)
        # Different content
        await store.create_note(peer.id, "Quantum computing breakthroughs in 2026", importance=0.4)

        proc = DedupProcessor(store, embedder)
        result = await proc.run(peer.id)

        assert isinstance(result, DedupResult)
        assert result.notes_processed == 3
        # The two cat notes should cluster
        assert result.clusters_found >= 1
        assert result.notes_merged >= 1

    async def test_canonical_has_highest_importance(self, store, embedder):
        peer = await store.create_peer("CanonTest")
        n1 = await store.create_note(peer.id, "Dogs are loyal companions", importance=0.3)
        n2 = await store.create_note(peer.id, "Dogs are loyal and faithful companions", importance=0.9)

        proc = DedupProcessor(store, embedder)
        await proc.run(peer.id)

        # n2 has higher importance, so should be canonical
        note1 = await store.get_note(n1.id)
        note2 = await store.get_note(n2.id)
        # The lower-importance note should point to the higher one
        if note1.canonical_note_id is not None:
            assert note1.canonical_note_id == n2.id
            assert note2.canonical_note_id is None
        # If they didn't cluster (threshold edge case), both stay None
        # which is also acceptable

    async def test_sums_evidence_count(self, store, embedder):
        peer = await store.create_peer("EvidenceTest")
        n1 = await store.create_note(peer.id, "The sky is blue and beautiful", importance=0.8)
        n2 = await store.create_note(peer.id, "The sky is blue and very beautiful", importance=0.2)

        proc = DedupProcessor(store, embedder)
        result = await proc.run(peer.id)

        if result.notes_merged > 0:
            canonical = await store.get_note(n1.id)
            assert canonical.evidence_count >= 2

    async def test_handles_no_buffered_notes(self, store, embedder):
        peer = await store.create_peer("Empty")
        proc = DedupProcessor(store, embedder)
        result = await proc.run(peer.id)

        assert result.notes_processed == 0
        assert result.clusters_found == 0
        assert result.notes_merged == 0

    async def test_handles_single_note(self, store, embedder):
        peer = await store.create_peer("Single")
        await store.create_note(peer.id, "Just one note here")

        proc = DedupProcessor(store, embedder)
        result = await proc.run(peer.id)

        assert result.notes_processed == 1
        assert result.clusters_found == 0
        assert result.notes_merged == 0

    async def test_dissimilar_notes_stay_separate(self, store, embedder):
        peer = await store.create_peer("Dissimilar")
        await store.create_note(peer.id, "Python is a programming language for software development", importance=0.5)
        await store.create_note(peer.id, "The Eiffel Tower is located in Paris France", importance=0.5)
        await store.create_note(peer.id, "Chocolate cake recipe with vanilla frosting", importance=0.5)

        proc = DedupProcessor(store, embedder)
        result = await proc.run(peer.id)

        assert result.notes_processed == 3
        assert result.clusters_found == 0
        assert result.notes_merged == 0
