"""Tests for A-MEM semantic linker."""

import pytest

from mnemosyne.db.sqlite_store import SQLiteStore
from mnemosyne.intelligence.linker import Linker
from mnemosyne.models import Note
from mnemosyne.vectors.embedder import Embedder
from mnemosyne.vectors.zvec_store import ZvecStore


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def embedder():
    """Module-scoped embedder (expensive model load)."""
    return Embedder()


@pytest.fixture
def zvec(tmp_path):
    """Function-scoped ZvecStore in a temp directory."""
    return ZvecStore(str(tmp_path))


@pytest.fixture
async def linker(store, zvec, embedder):
    """Function-scoped Linker with test stores."""
    return Linker(db=store, zvec=zvec, embedder=embedder)


# ── Helpers ──────────────────────────────────────────────────────────


async def _create_note_with_embedding(
    store: SQLiteStore,
    zvec: ZvecStore,
    embedder: Embedder,
    peer_id: str,
    content: str,
) -> tuple[Note, list[float]]:
    """Create a note in SQLite, embed it, insert to Zvec, return (note, embedding)."""
    note = await store.create_note(peer_id=peer_id, content=content)
    embedding = embedder.embed_document(content)
    zvec.insert(note.id, embedding)
    return note, embedding


# ── Tests ────────────────────────────────────────────────────────────


class TestLinker:
    async def test_creates_semantic_links(self, store, zvec, embedder, linker):
        """Similar notes about dogs get linked with type='semantic' and strength >= 0.75."""
        peer = await store.create_peer("LinkPeer1")

        n1, e1 = await _create_note_with_embedding(
            store, zvec, embedder, peer.id,
            "My golden retriever loves to play fetch in the park"
        )
        n2, e2 = await _create_note_with_embedding(
            store, zvec, embedder, peer.id,
            "Golden retrievers are the best dogs for families"
        )
        n3, e3 = await _create_note_with_embedding(
            store, zvec, embedder, peer.id,
            "I take my retriever to the dog park every weekend"
        )

        links = await linker.generate_links(n1, e1)

        assert len(links) > 0
        assert all(link.link_type == "semantic" for link in links)
        assert all(link.strength >= 0.75 for link in links)

    async def test_no_links_below_threshold(self, store, zvec, embedder, linker):
        """Unrelated notes produce no links."""
        peer = await store.create_peer("LinkPeer2")

        n1, e1 = await _create_note_with_embedding(
            store, zvec, embedder, peer.id,
            "Quantum entanglement in particle physics experiments"
        )
        n2, e2 = await _create_note_with_embedding(
            store, zvec, embedder, peer.id,
            "How to make a perfect sourdough bread recipe"
        )

        links = await linker.generate_links(n1, e1)
        assert links == []

    async def test_self_link_excluded(self, store, zvec, embedder, linker):
        """A note does not link to itself."""
        peer = await store.create_peer("LinkPeer3")

        n1, e1 = await _create_note_with_embedding(
            store, zvec, embedder, peer.id,
            "Dogs are wonderful companions"
        )

        links = await linker.generate_links(n1, e1)

        for link in links:
            assert link.source_note_id != link.target_note_id

    async def test_peer_isolation(self, store, zvec, embedder, linker):
        """Similar notes from different peers do not get linked."""
        peer_a = await store.create_peer("LinkPeerA")
        peer_b = await store.create_peer("LinkPeerB")

        n1, e1 = await _create_note_with_embedding(
            store, zvec, embedder, peer_a.id,
            "My cat loves sleeping on the couch all day long"
        )
        n2, e2 = await _create_note_with_embedding(
            store, zvec, embedder, peer_b.id,
            "Cats enjoy napping on soft couches and furniture"
        )

        links = await linker.generate_links(n1, e1)
        # No links should cross peer boundaries
        for link in links:
            target_note = await store.get_note(link.target_note_id)
            assert target_note.peer_id == peer_a.id

    async def test_duplicate_link_skipped(self, store, zvec, embedder, linker):
        """Calling generate_links twice does not error or create duplicates."""
        peer = await store.create_peer("LinkPeer5")

        n1, e1 = await _create_note_with_embedding(
            store, zvec, embedder, peer.id,
            "Tennis is a great racket sport for fitness"
        )
        n2, e2 = await _create_note_with_embedding(
            store, zvec, embedder, peer.id,
            "Playing tennis regularly improves cardiovascular health"
        )

        links1 = await linker.generate_links(n1, e1)
        links2 = await linker.generate_links(n1, e1)

        # Second call should not raise and should return empty (duplicates skipped)
        all_links = await store.get_links(n1.id)
        # Count unique (source, target, type) combos
        seen = set()
        for link in all_links:
            key = (link.source_note_id, link.target_note_id, link.link_type)
            assert key not in seen, f"Duplicate link found: {key}"
            seen.add(key)

    async def test_zvec_failure_graceful(self, store, zvec, embedder, linker, monkeypatch):
        """Zvec search failure returns [] without crashing."""
        peer = await store.create_peer("LinkPeer6")
        n1 = await store.create_note(peer_id=peer.id, content="Some note content")
        embedding = embedder.embed_document("Some note content")

        def broken_search(*args, **kwargs):
            raise RuntimeError("zvec exploded")

        monkeypatch.setattr(zvec, "search", broken_search)

        links = await linker.generate_links(n1, embedding)
        assert links == []

    async def test_find_neighbors(self, store, zvec, embedder, linker):
        """find_neighbors returns notes sorted by link strength descending."""
        peer = await store.create_peer("LinkPeer7")

        n1 = await store.create_note(peer_id=peer.id, content="Central note")
        n2 = await store.create_note(peer_id=peer.id, content="Weakly related")
        n3 = await store.create_note(peer_id=peer.id, content="Strongly related")

        await store.create_link(
            source_note_id=n1.id, target_note_id=n2.id,
            link_type="semantic", strength=0.6,
        )
        await store.create_link(
            source_note_id=n1.id, target_note_id=n3.id,
            link_type="semantic", strength=0.95,
        )

        neighbors = await linker.find_neighbors(n1.id)

        assert len(neighbors) == 2
        assert neighbors[0][0].id == n3.id
        assert neighbors[0][1] == 0.95
        assert neighbors[1][0].id == n2.id
        assert neighbors[1][1] == 0.6
