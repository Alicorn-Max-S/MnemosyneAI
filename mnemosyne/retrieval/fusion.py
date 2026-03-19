"""Reciprocal Rank Fusion and MMR deduplication for Mnemosyne.

Pure functions with no store dependencies.
"""

import logging

import numpy as np

from mnemosyne.config import MMR_SIMILARITY_THRESHOLD, RRF_K

logger = logging.getLogger(__name__)


def rrf_fuse(
    ranked_lists: list[list[str]],
    k: int = RRF_K,
) -> dict[str, float]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    For each list, each item at rank i (0-indexed) receives a score
    of 1/(k + i). Scores accumulate across lists.

    Returns dict mapping id to aggregated RRF score.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for i, item_id in enumerate(ranked_list):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + i)
    return scores


def mmr_dedup(
    scored_ids: list[str],
    embeddings: dict[str, list[float]],
    threshold: float = MMR_SIMILARITY_THRESHOLD,
) -> list[str]:
    """Remove near-duplicate results using Maximal Marginal Relevance.

    Walks scored_ids in order (assumed pre-sorted by score descending).
    Candidates without embeddings are auto-accepted. Candidates whose
    cosine similarity to any already-selected item exceeds the threshold
    are skipped.

    Returns filtered list preserving insertion order.
    """
    selected: list[str] = []
    selected_embeddings: list[np.ndarray] = []

    for item_id in scored_ids:
        raw_emb = embeddings.get(item_id)
        if raw_emb is None:
            selected.append(item_id)
            continue

        candidate = np.asarray(raw_emb, dtype=np.float64)
        norm = np.linalg.norm(candidate)
        if norm > 0:
            candidate = candidate / norm

        is_duplicate = False
        for sel_emb in selected_embeddings:
            similarity = float(np.dot(candidate, sel_emb))
            if similarity > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            selected.append(item_id)
            selected_embeddings.append(candidate)

    return selected
