"""Retrieval scoring and fusion for Mnemosyne Phase 3."""

from mnemosyne.retrieval.retriever import Retriever


def create_retriever(sqlite_store, zvec_store, embedder) -> Retriever:
    """Factory to create a Retriever wired to the given stores."""
    return Retriever(sqlite_store, zvec_store, embedder)
