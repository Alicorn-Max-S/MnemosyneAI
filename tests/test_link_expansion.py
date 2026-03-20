"""Tests for link expansion in the retrieval pipeline."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from mnemosyne.retrieval.retriever import Retriever


# ── Helpers ──────────────────────────────────────────────────────


def _make_orthogonal_vectors(n: int, dim: int = 384) -> list[list[float]]:
    """Generate n distinct vectors that won't trigger dedup."""
    rng = np.random.RandomState(99)
    vecs = []
    for _ in range(n):
        v = rng.randn(dim).astype(np.float64)
        v = v / np.linalg.norm(v)
        vecs.append(v.tolist())
    return vecs


_VECTORS = _make_orthogonal_vectors(30)
_VECTOR_INDEX = 0


def _next_vector() -> list[float]:
    global _VECTOR_INDEX
    v = _VECTORS[_VECTOR_INDEX % len(_VECTORS)]
    _VECTOR_INDEX += 1
    return v


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_vector_index():
    global _VECTOR_INDEX
    _VECTOR_INDEX = 0


@pytest.fixture
def mock_embedder():
    """Embedder returning distinct 384-dim vectors per call."""
    emb = MagicMock()
    emb.embed_query.side_effect = lambda q: _next_vector()
    emb.embed_document.side_effect = lambda t: _next_vector()
    emb.embed_documents.side_effect = lambda texts: [_next_vector() for _ in texts]
    return emb


@pytest.fixture
def mock_zvec():
    """ZvecStore with controllable search results."""
    zv = MagicMock()
    zv.search.return_value = []
    return zv


@pytest.fixture
def retriever(store, mock_zvec, mock_embedder):
    """Retriever wired to real SQLite store and mocked vector/embed."""
    return Retriever(store, mock_zvec, mock_embedder)


async def _create_peer_and_notes(store, peer_name, contents, **kwargs):
    """Create a peer and insert notes, return (peer, notes)."""
    peer = await store.create_peer(peer_name)
    notes = []
    for content in contents:
        note = await store.create_note(peer_id=peer.id, content=content, **kwargs)
        notes.append(note)
    return peer, notes


# ── Tests ────────────────────────────────────────────────────────


class TestLinkExpansion:
    @pytest.mark.asyncio
    async def test_linked_notes_added_to_results(self, store, mock_zvec, mock_embedder):
        """Linked neighbors appear in retrieval results."""
        peer, notes = await _create_peer_and_notes(
            store, "link_alice",
            ["dog training tips", "cat food brands", "bird watching guide"],
        )
        # Create a link: notes[0] -> notes[2]
        await store.create_link(
            source_note_id=notes[0].id,
            target_note_id=notes[2].id,
            link_type="semantic",
            strength=0.9,
        )

        # FTS finds notes[0], Zvec finds nothing extra
        mock_zvec.search.return_value = [{"id": notes[0].id, "score": 0.8}]

        retriever = Retriever(store, mock_zvec, mock_embedder)
        results = await retriever.retrieve("dog training", peer.id, limit=10)

        result_ids = {r.note.id for r in results}
        # notes[0] from direct search, notes[2] from link expansion
        assert notes[0].id in result_ids
        assert notes[2].id in result_ids

    @pytest.mark.asyncio
    async def test_expansion_respects_depth(self, store, mock_zvec, mock_embedder):
        """Depth=1 only returns direct neighbors, not 2-hop."""
        peer, notes = await _create_peer_and_notes(
            store, "link_bob",
            ["note A searchable", "note B middle", "note C distant"],
        )
        # A -> B -> C chain
        await store.create_link(
            source_note_id=notes[0].id, target_note_id=notes[1].id,
            link_type="semantic", strength=0.9,
        )
        await store.create_link(
            source_note_id=notes[1].id, target_note_id=notes[2].id,
            link_type="semantic", strength=0.9,
        )

        mock_zvec.search.return_value = [{"id": notes[0].id, "score": 0.8}]

        retriever = Retriever(store, mock_zvec, mock_embedder)
        results = await retriever.retrieve("searchable", peer.id, limit=10)

        result_ids = {r.note.id for r in results}
        assert notes[0].id in result_ids
        assert notes[1].id in result_ids  # direct neighbor
        assert notes[2].id not in result_ids  # 2-hop away

    @pytest.mark.asyncio
    async def test_expansion_respects_max(self, store, mock_zvec, mock_embedder):
        """Per-seed cap is respected: only LINK_EXPANSION_MAX neighbors per seed."""
        peer = await store.create_peer("link_carol")
        seed = await store.create_note(peer_id=peer.id, content="seed content findable")
        # Create 8 neighbors linked to the seed (more than LINK_EXPANSION_MAX=5)
        neighbors = []
        for i in range(8):
            n = await store.create_note(
                peer_id=peer.id, content=f"neighbor {i} unique content"
            )
            neighbors.append(n)
            await store.create_link(
                source_note_id=seed.id, target_note_id=n.id,
                link_type="semantic", strength=0.8,
            )

        mock_zvec.search.return_value = [{"id": seed.id, "score": 0.9}]

        retriever = Retriever(store, mock_zvec, mock_embedder)
        results = await retriever.retrieve("findable", peer.id, limit=20)

        # Seed + at most 5 neighbors = max 6
        link_results = [r for r in results if "link" in r.source]
        assert len(link_results) <= 5

    @pytest.mark.asyncio
    async def test_no_links_no_expansion(self, store, mock_zvec, mock_embedder):
        """No links → results unchanged from Phase 3 behavior."""
        peer, notes = await _create_peer_and_notes(
            store, "link_dave", ["standalone note searchterm"]
        )
        mock_zvec.search.return_value = [{"id": notes[0].id, "score": 0.8}]

        retriever = Retriever(store, mock_zvec, mock_embedder)
        results = await retriever.retrieve("searchterm", peer.id)

        assert len(results) >= 1
        # No link sources since no links exist
        for r in results:
            assert "link" not in r.source

    @pytest.mark.asyncio
    async def test_link_source_tag(self, store, mock_zvec, mock_embedder):
        """Link-expanded notes have source containing 'link'."""
        peer, notes = await _create_peer_and_notes(
            store, "link_eve",
            ["primary searchitem here", "linked neighbor content"],
        )
        await store.create_link(
            source_note_id=notes[0].id, target_note_id=notes[1].id,
            link_type="semantic", strength=0.85,
        )

        mock_zvec.search.return_value = [{"id": notes[0].id, "score": 0.8}]

        retriever = Retriever(store, mock_zvec, mock_embedder)
        results = await retriever.retrieve("searchitem", peer.id, limit=10)

        link_expanded = [r for r in results if r.note.id == notes[1].id]
        assert len(link_expanded) == 1
        assert "link" in link_expanded[0].source
