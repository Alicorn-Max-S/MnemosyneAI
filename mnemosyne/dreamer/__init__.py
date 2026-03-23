"""Dreamer — background processing for Mnemosyne."""

from mnemosyne.dreamer.dedup import DedupProcessor, DedupResult
from mnemosyne.dreamer.gemini_client import (
    GeminiBatchError,
    GeminiClient,
    GeminiError,
    GeminiTimeoutError,
)
from mnemosyne.dreamer.orchestrator import CycleResult, DreamerOrchestrator
from mnemosyne.dreamer.processor import DreamerProcessor
from mnemosyne.dreamer.task_builder import (
    build_contradiction_requests,
    build_link_requests,
    build_pattern_requests,
    build_profile_request,
)

__all__ = [
    "CycleResult",
    "DedupProcessor",
    "DedupResult",
    "DreamerOrchestrator",
    "DreamerProcessor",
    "GeminiBatchError",
    "GeminiClient",
    "GeminiError",
    "GeminiTimeoutError",
    "build_contradiction_requests",
    "build_link_requests",
    "build_pattern_requests",
    "build_profile_request",
    "create_dreamer",
]


def create_dreamer(
    db, embedder, zvec, gemini_client, deriver
) -> DreamerOrchestrator:
    """Factory to create a DreamerOrchestrator with all dependencies."""
    return DreamerOrchestrator(
        db=db,
        embedder=embedder,
        zvec=zvec,
        gemini_client=gemini_client,
        deriver=deriver,
    )
