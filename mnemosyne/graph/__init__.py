"""MAGMA multi-graph module for entity extraction and co-occurrence analysis."""

from mnemosyne.graph.magma import MAGMAGraph

__all__ = ["MAGMAGraph"]


def create_magma_graph(db):
    """Factory to create a MAGMAGraph instance."""
    return MAGMAGraph(db)
