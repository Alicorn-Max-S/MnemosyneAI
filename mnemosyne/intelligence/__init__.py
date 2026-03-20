"""Intelligence layer: ColBERT reranking, A-MEM semantic linking, and static profiles."""

from mnemosyne.intelligence.linker import Linker
from mnemosyne.intelligence.profiler import Profiler
from mnemosyne.intelligence.reranker import ColBERTReranker


def create_linker(db, zvec, embedder) -> Linker:
    """Factory to create a Linker wired to the given stores."""
    return Linker(db=db, zvec=zvec, embedder=embedder)


def create_reranker() -> ColBERTReranker:
    """Factory to create a ColBERTReranker with default settings."""
    return ColBERTReranker()


def create_profiler(db, deriver) -> Profiler:
    """Factory to create a Profiler wired to the given stores."""
    return Profiler(db=db, deriver=deriver)


__all__ = [
    "ColBERTReranker", "Linker", "Profiler",
    "create_linker", "create_profiler", "create_reranker",
]
