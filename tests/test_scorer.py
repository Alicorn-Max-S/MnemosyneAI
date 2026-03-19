"""Tests for mnemosyne.retrieval.scorer — pure scoring functions."""

import pytest

from mnemosyne.config import (
    DECAY_HIGH_IMPORTANCE_FLOOR,
    INFERENCE_DISCOUNT,
    PROVENANCE_WEIGHTS,
)
from mnemosyne.retrieval.scorer import (
    compute_composite_score,
    compute_decay_strength,
    compute_inference_discount,
    compute_provenance_weight,
    compute_surfacing_fatigue,
)


class TestComputeDecayStrength:
    """Tests for Ebbinghaus decay strength computation."""

    def test_decay_disabled_below_threshold(self):
        """Below 100 memories, decay is disabled — always returns 1.0."""
        assert compute_decay_strength(0.5, 30, 5, 50) == 1.0
        assert compute_decay_strength(0.1, 100, 0, 99) == 1.0
        assert compute_decay_strength(0.9, 365, 10, 0) == 1.0

    def test_decay_ramp_at_threshold(self):
        """At exactly 100 memories, ramp is 0.1 — very little decay."""
        strength_100 = compute_decay_strength(0.5, 30, 0, 100)
        strength_1000 = compute_decay_strength(0.5, 30, 0, 1000)
        # At 100 memories, decay is much weaker than at 1000
        assert strength_100 > strength_1000

    def test_decay_ramp_progression(self):
        """Decay increases as total_memories grows from 100 to 1000."""
        strengths = [
            compute_decay_strength(0.5, 30, 0, mem)
            for mem in [200, 400, 600, 800, 1000]
        ]
        # Each should be less than or equal to the previous (more decay)
        for i in range(1, len(strengths)):
            assert strengths[i] <= strengths[i - 1]

    def test_high_importance_floor(self):
        """High-importance memories never drop below the floor."""
        strength = compute_decay_strength(0.8, 9999, 0, 1000)
        assert strength >= DECAY_HIGH_IMPORTANCE_FLOOR

    def test_high_importance_floor_boundary(self):
        """Importance at exactly 0.7 triggers the floor."""
        strength = compute_decay_strength(0.7, 9999, 0, 1000)
        assert strength >= DECAY_HIGH_IMPORTANCE_FLOOR

    def test_below_high_importance_no_floor(self):
        """Importance below 0.7 does NOT get the floor protection."""
        strength = compute_decay_strength(0.69, 9999, 0, 1000)
        assert strength < DECAY_HIGH_IMPORTANCE_FLOOR

    def test_zero_importance(self):
        """Zero importance yields zero strength."""
        assert compute_decay_strength(0.0, 30, 5, 500) == 0.0

    def test_zero_days_since_access(self):
        """Just-accessed memory: exp(0) = 1, strength based on importance and access."""
        strength = compute_decay_strength(0.5, 0, 3, 500)
        expected = 0.5 * 1.0 * (1 + 0.15 * 3)  # importance * exp(0) * access_boost
        assert abs(strength - expected) < 1e-9

    def test_access_count_boosts_strength(self):
        """More accesses increase strength."""
        low_access = compute_decay_strength(0.5, 10, 1, 500)
        high_access = compute_decay_strength(0.5, 10, 20, 500)
        assert high_access > low_access

    def test_strength_never_negative(self):
        """Strength is clamped to >= 0.0."""
        strength = compute_decay_strength(0.0, 100000, 0, 2000)
        assert strength >= 0.0


class TestComputeProvenanceWeight:
    """Tests for provenance weight lookup."""

    def test_all_known_provenances(self):
        """Each known provenance returns its configured weight."""
        for key, expected in PROVENANCE_WEIGHTS.items():
            assert compute_provenance_weight(key) == expected

    def test_unknown_provenance_returns_default(self):
        """Unknown provenance returns 0.5."""
        assert compute_provenance_weight("unknown_source") == 0.5
        assert compute_provenance_weight("") == 0.5


class TestComputeSurfacingFatigue:
    """Tests for surfacing fatigue discount."""

    def test_zero_surfacings(self):
        """Never surfaced → full weight."""
        assert compute_surfacing_fatigue(0) == 1.0

    def test_monotonically_decreasing(self):
        """Fatigue strictly decreases as times_surfaced increases."""
        values = [compute_surfacing_fatigue(i) for i in range(11)]
        for i in range(1, len(values)):
            assert values[i] < values[i - 1]

    def test_known_values(self):
        """Check a few specific values."""
        assert abs(compute_surfacing_fatigue(1) - 1.0 / 1.1) < 1e-9
        assert abs(compute_surfacing_fatigue(2) - 1.0 / 1.2) < 1e-9
        assert abs(compute_surfacing_fatigue(10) - 1.0 / 2.0) < 1e-9

    def test_always_positive(self):
        """Fatigue is always > 0."""
        assert compute_surfacing_fatigue(1000000) > 0.0


class TestComputeInferenceDiscount:
    """Tests for inference discount by note type."""

    def test_observation_full_weight(self):
        assert compute_inference_discount("observation") == 1.0

    def test_inference_discounted(self):
        assert compute_inference_discount("inference") == INFERENCE_DISCOUNT

    def test_unknown_type_full_weight(self):
        assert compute_inference_discount("something_else") == 1.0
        assert compute_inference_discount("") == 1.0


class TestComputeCompositeScore:
    """Tests for composite score multiplication."""

    def test_all_ones(self):
        """All factors at 1.0 → score equals rrf_score."""
        assert compute_composite_score(0.5, 1.0, 1.0, 1.0, 1.0) == 0.5

    def test_product_of_factors(self):
        """Composite is the product of all inputs."""
        result = compute_composite_score(0.8, 0.9, 0.7, 0.6, 0.5)
        expected = 0.8 * 0.9 * 0.7 * 0.6 * 0.5
        assert abs(result - expected) < 1e-9

    def test_zero_rrf_score(self):
        """Zero RRF score → zero composite regardless of other factors."""
        assert compute_composite_score(0.0, 0.9, 0.8, 0.7, 0.6) == 0.0

    def test_zero_any_factor(self):
        """Any zero factor → zero composite."""
        assert compute_composite_score(0.5, 0.0, 0.8, 0.7, 0.6) == 0.0
        assert compute_composite_score(0.5, 0.8, 0.0, 0.7, 0.6) == 0.0
        assert compute_composite_score(0.5, 0.8, 0.7, 0.0, 0.6) == 0.0
        assert compute_composite_score(0.5, 0.8, 0.7, 0.6, 0.0) == 0.0
