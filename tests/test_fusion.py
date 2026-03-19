"""Tests for mnemosyne.retrieval.fusion — RRF and MMR deduplication."""

import numpy as np
import pytest

from mnemosyne.config import MMR_SIMILARITY_THRESHOLD, RRF_K
from mnemosyne.retrieval.fusion import mmr_dedup, rrf_fuse


class TestRrfFuse:
    """Tests for Reciprocal Rank Fusion."""

    def test_overlap_scores_higher(self):
        """Item appearing in both lists scores higher than item in one."""
        lists = [["a", "b", "c"], ["a", "d", "e"]]
        scores = rrf_fuse(lists)
        # "a" is rank 0 in both lists
        assert scores["a"] > scores["b"]
        assert scores["a"] > scores["d"]

    def test_empty_input(self):
        """Empty input returns empty dict."""
        assert rrf_fuse([]) == {}
        assert rrf_fuse([[]]) == {}

    def test_rank_order_matters(self):
        """Rank 0 contributes more than rank 5."""
        lists = [["high", "x", "x2", "x3", "x4", "low"]]
        scores = rrf_fuse(lists)
        assert scores["high"] > scores["low"]

    def test_single_list(self):
        """Single list produces decreasing scores by rank."""
        lists = [["a", "b", "c"]]
        scores = rrf_fuse(lists)
        assert scores["a"] > scores["b"] > scores["c"]

    def test_score_formula(self):
        """Verify exact RRF score computation."""
        lists = [["a", "b"], ["b", "a"]]
        scores = rrf_fuse(lists)
        # a: 1/(60+0) + 1/(60+1) = 1/60 + 1/61
        expected_a = 1.0 / RRF_K + 1.0 / (RRF_K + 1)
        # b: 1/(60+1) + 1/(60+0) = same
        expected_b = 1.0 / (RRF_K + 1) + 1.0 / RRF_K
        assert abs(scores["a"] - expected_a) < 1e-12
        assert abs(scores["b"] - expected_b) < 1e-12

    def test_custom_k(self):
        """Custom k parameter changes scores."""
        lists = [["a", "b"]]
        scores_default = rrf_fuse(lists)
        scores_custom = rrf_fuse(lists, k=1)
        # With k=1, rank 0 gets 1/1 = 1.0 instead of 1/60
        assert scores_custom["a"] > scores_default["a"]

    def test_three_lists(self):
        """Item in all three lists scores highest."""
        lists = [["a", "b"], ["a", "c"], ["a", "d"]]
        scores = rrf_fuse(lists)
        assert scores["a"] > scores["b"]
        assert scores["a"] > scores["c"]
        assert scores["a"] > scores["d"]


class TestMmrDedup:
    """Tests for MMR-based deduplication."""

    def test_identical_embeddings_first_only(self):
        """Two items with identical embeddings — only the first is kept."""
        emb = [1.0, 0.0, 0.0]
        embeddings = {"a": emb, "b": emb}
        result = mmr_dedup(["a", "b"], embeddings)
        assert result == ["a"]

    def test_orthogonal_all_kept(self):
        """Orthogonal embeddings all pass — no similarity."""
        embeddings = {
            "a": [1.0, 0.0, 0.0],
            "b": [0.0, 1.0, 0.0],
            "c": [0.0, 0.0, 1.0],
        }
        result = mmr_dedup(["a", "b", "c"], embeddings)
        assert result == ["a", "b", "c"]

    def test_threshold_boundary_kept(self):
        """Cosine exactly at threshold is NOT filtered (uses > not >=)."""
        # Build two unit vectors with cosine similarity exactly at threshold
        cos_val = MMR_SIMILARITY_THRESHOLD
        sin_val = np.sqrt(1 - cos_val ** 2)
        emb_a = [cos_val, sin_val, 0.0]
        emb_b = [1.0, 0.0, 0.0]
        embeddings = {"a": emb_a, "b": emb_b}
        result = mmr_dedup(["b", "a"], embeddings)
        # Similarity is exactly threshold, not greater — both kept
        assert result == ["b", "a"]

    def test_above_threshold_filtered(self):
        """Cosine above threshold IS filtered."""
        # Very similar vectors
        emb_a = [1.0, 0.0, 0.0]
        emb_b = [0.999, 0.001, 0.0]  # Nearly identical direction
        embeddings = {"a": emb_a, "b": emb_b}
        result = mmr_dedup(["a", "b"], embeddings)
        assert result == ["a"]

    def test_missing_embeddings_auto_accepted(self):
        """Items without embeddings always pass dedup."""
        embeddings = {"a": [1.0, 0.0, 0.0]}
        result = mmr_dedup(["a", "no_emb_1", "no_emb_2"], embeddings)
        assert result == ["a", "no_emb_1", "no_emb_2"]

    def test_order_preserved(self):
        """Output order matches input order for accepted items."""
        embeddings = {
            "x": [1.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
            "z": [0.0, 0.0, 1.0],
        }
        result = mmr_dedup(["x", "y", "z"], embeddings)
        assert result == ["x", "y", "z"]

    def test_empty_input(self):
        """Empty scored_ids returns empty list."""
        assert mmr_dedup([], {}) == []

    def test_all_missing_embeddings(self):
        """If no embeddings provided, all items are accepted."""
        result = mmr_dedup(["a", "b", "c"], {})
        assert result == ["a", "b", "c"]
