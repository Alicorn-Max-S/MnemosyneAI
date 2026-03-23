"""Batch deduplication of buffered notes using cosine similarity and union-find."""

import asyncio
import logging
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from mnemosyne.config import DEDUP_COSINE_THRESHOLD, DEDUP_MIN_CLUSTER_SIZE
from mnemosyne.db.sqlite_store import SQLiteStore
from mnemosyne.vectors.embedder import Embedder

logger = logging.getLogger(__name__)


@dataclass
class DedupResult:
    """Result of a batch dedup run."""

    notes_processed: int
    clusters_found: int
    notes_merged: int


class _UnionFind:
    """Disjoint-set with path compression and union by rank."""

    def __init__(self, elements: list[str]) -> None:
        """Initialize each element as its own root."""
        self._parent: dict[str, str] = {e: e for e in elements}
        self._rank: dict[str, int] = {e: 0 for e in elements}

    def find(self, x: str) -> str:
        """Find root of x with path compression."""
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        """Merge sets containing x and y by rank."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


class DedupProcessor:
    """Deduplicates buffered notes for a peer using embedding cosine similarity."""

    def __init__(self, db: SQLiteStore, embedder: Embedder) -> None:
        """Initialize with database and embedder references."""
        self._db = db
        self._embedder = embedder

    async def run(self, peer_id: str) -> DedupResult:
        """Run dedup on all buffered notes for a peer.

        Returns a DedupResult summarizing what happened.
        """
        notes = await self._db.get_buffered_notes(peer_id)
        if len(notes) <= 1:
            return DedupResult(notes_processed=len(notes), clusters_found=0, notes_merged=0)

        texts = [n.content for n in notes]
        embeddings = await asyncio.to_thread(self._embedder.embed_documents, texts)
        emb = np.array(embeddings)

        # Cosine similarity matrix (embeddings are already L2-normalized)
        sim_matrix = np.dot(emb, emb.T)

        # Union-find clustering
        ids = [n.id for n in notes]
        uf = _UnionFind(ids)
        n = len(ids)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= DEDUP_COSINE_THRESHOLD:
                    uf.union(ids[i], ids[j])

        # Group by root
        clusters: dict[str, list[int]] = defaultdict(list)
        for idx, nid in enumerate(ids):
            clusters[uf.find(nid)].append(idx)

        # Filter to clusters meeting minimum size
        merge_clusters = [
            indices for indices in clusters.values()
            if len(indices) >= DEDUP_MIN_CLUSTER_SIZE
        ]

        total_merged = 0
        for indices in merge_clusters:
            cluster_notes = [notes[i] for i in indices]

            # Pick highest-importance note as canonical
            canonical = max(cluster_notes, key=lambda n: n.importance)
            merged = [n for n in cluster_notes if n.id != canonical.id]
            merged_ids = [n.id for n in merged]

            await self._db.merge_notes(canonical.id, merged_ids)

            # Compute unique sessions across the whole cluster
            all_cluster_ids = [n.id for n in cluster_notes]
            unique_sessions = await self._db.get_unique_sessions_for_notes(all_cluster_ids)

            # Recompute importance
            importance = (
                canonical.emotional_weight * 0.6
                + (1 - math.exp(-0.15 * unique_sessions)) * 0.4
            )
            await self._db.update_note(canonical.id, importance=importance)

            total_merged += len(merged_ids)
            logger.info(
                "Dedup cluster: canonical=%s, merged=%d notes",
                canonical.id,
                len(merged_ids),
            )

        return DedupResult(
            notes_processed=len(notes),
            clusters_found=len(merge_clusters),
            notes_merged=total_merged,
        )
