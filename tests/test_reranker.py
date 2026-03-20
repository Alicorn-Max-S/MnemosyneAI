"""Tests for ColBERT reranker."""

import pytest

from mnemosyne.intelligence.reranker import ColBERTReranker


@pytest.fixture(scope="module")
def reranker():
    """Module-scoped reranker (expensive model load)."""
    return ColBERTReranker()


class TestColBERTReranker:
    def test_reranks_candidates(self, reranker):
        """Reranker returns reordered (id, score) tuples for multiple candidates."""
        candidates = [
            ("n1", "The weather is sunny today"),
            ("n2", "Dogs are loyal pets"),
            ("n3", "Python is a programming language"),
            ("n4", "Cats love to nap"),
            ("n5", "The stock market crashed"),
        ]
        results = reranker.rerank("household pets", candidates)

        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        ids = [r[0] for r in results]
        assert set(ids).issubset({"n1", "n2", "n3", "n4", "n5"})

    def test_single_candidate_passthrough(self, reranker):
        """Single candidate returns exactly 1 result with matching ID."""
        candidates = [("note_abc", "The quick brown fox")]
        results = reranker.rerank("fox", candidates)

        assert len(results) == 1
        assert results[0][0] == "note_abc"

    def test_empty_candidates(self, reranker):
        """Empty candidates list returns empty list."""
        results = reranker.rerank("any query", [])
        assert results == []

    def test_top_n_truncation(self, reranker):
        """Results truncated to top_n."""
        candidates = [(f"n{i}", f"Document number {i} about various topics") for i in range(10)]
        results = reranker.rerank("topics", candidates, top_n=3)

        assert len(results) == 3

    def test_scores_are_floats(self, reranker):
        """All scores are float type."""
        candidates = [
            ("n1", "Apples are red fruit"),
            ("n2", "Bananas are yellow"),
        ]
        results = reranker.rerank("fruit colors", candidates)

        assert all(isinstance(score, float) for _, score in results)

    def test_relevant_ranked_higher(self, reranker):
        """Pet-related candidates ranked in top 3 for a pet query."""
        candidates = [
            ("n1", "The theory of relativity was proposed by Einstein"),
            ("n2", "Cats and dogs make wonderful household pets"),
            ("n3", "SQL databases use structured query language"),
            ("n4", "The Eiffel Tower is in Paris"),
            ("n5", "Quantum computing uses qubits"),
        ]
        results = reranker.rerank("cats and dogs", candidates, top_n=5)

        top_3_ids = [r[0] for r in results[:3]]
        assert "n2" in top_3_ids
