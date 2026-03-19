"""Integration tests for MemoryAPI coordination layer."""

import pytest

from mnemosyne.api.memory_api import MemoryAPI
from mnemosyne.models import Note
from mnemosyne.vectors.embedder import Embedder


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    """Module-scoped embedder — avoid reloading the 500MB model per test."""
    return Embedder()


@pytest.fixture
async def api(tmp_path, embedder):
    """Function-scoped MemoryAPI backed by a temp directory and shared embedder."""
    m = MemoryAPI(data_dir=str(tmp_path), embedder=embedder)
    await m.initialize()
    yield m
    await m.close()


# ── Round-trip ─────────────────────────────────────────────────


async def test_full_round_trip(api: MemoryAPI) -> None:
    """add_note → keyword, vector, hybrid search all find the note."""
    peer = await api.create_peer("Alice")
    session = await api.start_session(peer.id)
    await api.add_message(session.id, peer.id, "user", "Tell me about my cat.")
    note = await api.add_note(
        peer.id,
        "Max has a cat named Buddy",
        session_id=session.id,
        keywords=["cat", "Buddy"],
        tags=["pet"],
    )

    # keyword
    kw = await api.search_keyword("cat", peer.id)
    assert any(n.id == note.id for n in kw)

    # vector
    vec = await api.search_vector("What pets does the user have?", peer.id)
    assert any(n.id == note.id for n, _ in vec)

    # hybrid
    hyb = await api.search_hybrid("Buddy the cat", peer.id)
    assert any(n.id == note.id for n, _ in hyb)


# ── Dual-store consistency ─────────────────────────────────────


async def test_note_in_sqlite_and_zvec(api: MemoryAPI) -> None:
    """After add_note the note exists in both SQLite and Zvec."""
    peer = await api.create_peer("Bob")
    note = await api.add_note(peer.id, "Bob loves hiking in the mountains")

    # SQLite
    fetched = await api.get_note(note.id)
    assert fetched is not None
    assert fetched.zvec_id == note.id

    # Zvec direct search
    vec = await api.search_vector("hiking mountains", peer.id)
    assert any(n.id == note.id for n, _ in vec)


# ── Hybrid ranking ─────────────────────────────────────────────


async def test_hybrid_ranks_most_relevant_first(api: MemoryAPI) -> None:
    """Hybrid search for pets puts pet-related notes near the top."""
    peer = await api.create_peer("Carol")
    await api.add_note(peer.id, "The GDP of France was 2.8 trillion dollars")
    await api.add_note(peer.id, "Python 3.12 was released in October 2023")
    pet_note = await api.add_note(
        peer.id,
        "Carol has two dogs and a parrot",
        keywords=["dogs", "parrot", "pets"],
        tags=["pets"],
    )
    await api.add_note(peer.id, "The recipe needs three cups of flour")
    await api.add_note(peer.id, "Quantum entanglement is non-local")

    results = await api.search_hybrid("What pets does Carol have?", peer.id, limit=3)
    top_ids = [n.id for n, _ in results]
    assert pet_note.id in top_ids


# ── Keyword search contract ────────────────────────────────────


async def test_keyword_returns_notes_only(api: MemoryAPI) -> None:
    """search_keyword returns list[Note], not tuples."""
    peer = await api.create_peer("Dan")
    await api.add_note(peer.id, "Dan plays guitar", keywords=["guitar"])

    results = await api.search_keyword("guitar", peer.id)
    assert len(results) >= 1
    assert all(isinstance(n, Note) for n in results)


async def test_keyword_no_results(api: MemoryAPI) -> None:
    """search_keyword for a nonexistent term returns empty list."""
    peer = await api.create_peer("Eve")
    await api.add_note(peer.id, "Eve studies mathematics")

    results = await api.search_keyword("xylophone", peer.id)
    assert results == []


# ── Peer filtering ─────────────────────────────────────────────


async def test_vector_filters_by_peer(api: MemoryAPI) -> None:
    """Vector search only returns the queried peer's notes."""
    peer_a = await api.create_peer("Fay")
    peer_b = await api.create_peer("Gus")

    await api.add_note(peer_a.id, "Fay enjoys surfing at the beach")
    await api.add_note(peer_b.id, "Gus enjoys surfing at the beach")

    results_a = await api.search_vector("surfing", peer_a.id)
    assert all(n.peer_id == peer_a.id for n, _ in results_a)

    results_b = await api.search_vector("surfing", peer_b.id)
    assert all(n.peer_id == peer_b.id for n, _ in results_b)


# ── Lifecycle ──────────────────────────────────────────────────


async def test_close_runs_clean(api: MemoryAPI) -> None:
    """close() completes without raising."""
    await api.close()


async def test_reopen_recovers_state(tmp_path, embedder) -> None:
    """Close and reinitialize with same data_dir — state persists."""
    data_dir = str(tmp_path)

    api = MemoryAPI(data_dir=data_dir, embedder=embedder)
    await api.initialize()
    peer = await api.create_peer("Hal")
    note = await api.add_note(peer.id, "Hal collects stamps", keywords=["stamps"])
    await api.close()

    # Reopen
    api2 = MemoryAPI(data_dir=data_dir, embedder=embedder)
    await api2.initialize()

    assert await api2.get_peer(peer.id) is not None
    assert await api2.get_note(note.id) is not None

    kw = await api2.search_keyword("stamps", peer.id)
    assert any(n.id == note.id for n in kw)

    vec = await api2.search_vector("stamp collecting", peer.id)
    assert any(n.id == note.id for n, _ in vec)
    await api2.close()


# ── Session lifecycle ──────────────────────────────────────────


async def test_start_and_end_session(api: MemoryAPI) -> None:
    """Session can be started and ended."""
    peer = await api.create_peer("Ivy")
    session = await api.start_session(peer.id)
    assert session.ended_at is None

    ended = await api.end_session(session.id)
    assert ended.ended_at is not None


# ── Message CRUD ───────────────────────────────────────────────


async def test_add_message(api: MemoryAPI) -> None:
    """Messages are persisted through the API."""
    peer = await api.create_peer("Jay")
    session = await api.start_session(peer.id)
    msg = await api.add_message(session.id, peer.id, "user", "Hello!")

    assert msg.content == "Hello!"
    assert msg.role == "user"
    assert msg.session_id == session.id
