"""Tests for the ZvecStore vector store."""

import random

import pytest

from mnemosyne.vectors.zvec_store import ZvecStore


def _random_embedding(dim: int = 384) -> list[float]:
    """Generate a random embedding vector."""
    return [random.gauss(0, 1) for _ in range(dim)]


@pytest.fixture
def store(tmp_path: str) -> ZvecStore:
    """Create a ZvecStore in a temp directory."""
    return ZvecStore(data_dir=str(tmp_path))


def test_insert_and_query(store: ZvecStore) -> None:
    """Insert one doc and query with the same embedding to find it."""
    emb = _random_embedding()
    store.insert("note_001", emb)

    results = store.search(emb, top_k=5)
    assert len(results) >= 1
    assert results[0]["id"] == "note_001"
    assert isinstance(results[0]["score"], float)


def test_batch_insert_and_search(store: ZvecStore) -> None:
    """Batch insert multiple docs and verify ranked results."""
    target = _random_embedding()
    # Insert the target and some noise
    items = [("target", target)]
    for i in range(5):
        items.append((f"noise_{i}", _random_embedding()))
    store.insert_batch(items)

    results = store.search(target, top_k=10)
    assert len(results) >= 1
    # The target should be the top result (closest to itself)
    assert results[0]["id"] == "target"


def test_delete_removes_from_results(store: ZvecStore) -> None:
    """After deleting a doc, it should not appear in search results."""
    emb = _random_embedding()
    store.insert("to_delete", emb)

    # Confirm it's found
    results = store.search(emb, top_k=5)
    found_ids = [r["id"] for r in results]
    assert "to_delete" in found_ids

    # Delete and re-query
    store.delete("to_delete")
    store.optimize()
    results = store.search(emb, top_k=5)
    found_ids = [r["id"] for r in results]
    assert "to_delete" not in found_ids


def test_empty_query(store: ZvecStore) -> None:
    """Querying an empty collection should return an empty list."""
    results = store.search(_random_embedding(), top_k=5)
    assert results == []


def test_stats(store: ZvecStore) -> None:
    """stats() should return a dict with path info."""
    info = store.stats()
    assert "path" in info
