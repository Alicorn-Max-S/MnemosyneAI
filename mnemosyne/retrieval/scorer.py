"""Pure scoring functions for Mnemosyne retrieval pipeline.

All functions are stateless and depend only on config constants.
Formulas from mnemosyne_architecture_spec.md.
"""

import logging
from math import exp

from mnemosyne.config import (
    DECAY_ACCESS_BOOST,
    DECAY_HIGH_IMPORTANCE_FLOOR,
    DECAY_HIGH_IMPORTANCE_THRESHOLD,
    DECAY_IMPORTANCE_FACTOR,
    DECAY_LAMBDA_BASE,
    DECAY_MEMORY_THRESHOLD,
    DECAY_RAMP_MAX,
    INFERENCE_DISCOUNT,
    PROVENANCE_WEIGHTS,
    SURFACING_FATIGUE_RATE,
)

logger = logging.getLogger(__name__)


def compute_decay_strength(
    importance: float,
    days_since_access: float,
    access_count: int,
    total_memories: int,
) -> float:
    """Compute Ebbinghaus-style memory decay strength.

    Returns 1.0 (no decay) when total_memories < DECAY_MEMORY_THRESHOLD.
    Ramps linearly from threshold to DECAY_RAMP_MAX. High-importance
    memories are protected by a floor value.
    """
    if total_memories < DECAY_MEMORY_THRESHOLD:
        return 1.0

    ramp = min(1.0, total_memories / DECAY_RAMP_MAX)
    lambda_eff = DECAY_LAMBDA_BASE * (1 - importance * DECAY_IMPORTANCE_FACTOR) * ramp

    strength = (
        importance
        * exp(-lambda_eff * days_since_access)
        * (1 + DECAY_ACCESS_BOOST * access_count)
    )

    if importance >= DECAY_HIGH_IMPORTANCE_THRESHOLD:
        strength = max(strength, DECAY_HIGH_IMPORTANCE_FLOOR)

    return max(strength, 0.0)


def compute_provenance_weight(provenance: str) -> float:
    """Return scoring weight for a given provenance type.

    Unknown provenance values return 0.5 as a defensive default.
    """
    return PROVENANCE_WEIGHTS.get(provenance, 0.5)


def compute_surfacing_fatigue(times_surfaced: int) -> float:
    """Compute surfacing fatigue discount.

    Monotonically decreasing: 1.0 at 0, ~0.91 at 1, ~0.83 at 2, etc.
    """
    return 1.0 / (1.0 + SURFACING_FATIGUE_RATE * times_surfaced)


def compute_inference_discount(note_type: str) -> float:
    """Return scoring discount based on note type.

    Observations get full weight (1.0), inferences are discounted.
    Unknown types default to 1.0.
    """
    if note_type == "inference":
        return INFERENCE_DISCOUNT
    return 1.0


def compute_composite_score(
    rrf_score: float,
    decay: float,
    provenance: float,
    fatigue: float,
    inference_discount: float,
) -> float:
    """Multiply all scoring factors into a single composite score."""
    return rrf_score * decay * provenance * fatigue * inference_discount
