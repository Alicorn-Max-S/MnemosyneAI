"""High-level MemoryAPI coordinating SQLite + Zvec."""

import asyncio
import logging
import os
import re

from mnemosyne.config import DEFAULT_DATA_DIR, RRF_K, SQLITE_DB_NAME
from mnemosyne.db.sqlite_store import SQLiteStore
from mnemosyne.models import Message, Note, Peer, RetrievalResult, Session
from mnemosyne.retrieval import create_retriever
from mnemosyne.vectors.embedder import Embedder
from mnemosyne.vectors.zvec_store import ZvecStore

logger = logging.getLogger(__name__)


class MemoryAPI:
    """Public coordination layer tying SQLite, Zvec, and Embedder together."""

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        embedder: Embedder | None = None,
    ) -> None:
        """Store configuration; call initialize() before use."""
        self.data_dir = data_dir
        self._sqlite: SQLiteStore | None = None
        self._zvec: ZvecStore | None = None
        self._embedder: Embedder | None = embedder
        self._retriever = None

    async def initialize(self) -> None:
        """Create stores, load embedder if not injected, and prepare for use."""
        os.makedirs(self.data_dir, exist_ok=True)

        self._sqlite = SQLiteStore(os.path.join(self.data_dir, SQLITE_DB_NAME))
        await self._sqlite.initialize()

        if self._embedder is None:
            self._embedder = await asyncio.to_thread(Embedder)

        self._zvec = await asyncio.to_thread(ZvecStore, self.data_dir)
        self._retriever = create_retriever(self._sqlite, self._zvec, self._embedder)
        logger.info("MemoryAPI initialized at %s", self.data_dir)

    async def close(self) -> None:
        """Shut down the SQLite connection and release references."""
        if self._sqlite:
            await self._sqlite.close()
        self._sqlite = None
        self._zvec = None
        self._embedder = None
        self._retriever = None

    # ── Delegated CRUD ─────────────────────────────────────────────

    async def create_peer(
        self, name: str, peer_type: str = "user"
    ) -> Peer:
        """Create a new peer."""
        return await self._sqlite.create_peer(name, peer_type=peer_type)

    async def get_peer(self, peer_id: str) -> Peer | None:
        """Get a peer by ID."""
        return await self._sqlite.get_peer(peer_id)

    async def start_session(self, peer_id: str) -> Session:
        """Start a new conversation session."""
        return await self._sqlite.create_session(peer_id)

    async def end_session(self, session_id: str) -> Session | None:
        """End a session."""
        return await self._sqlite.end_session(session_id)

    async def add_message(
        self,
        session_id: str,
        peer_id: str,
        role: str,
        content: str,
    ) -> Message:
        """Add a message to a session."""
        return await self._sqlite.add_message(session_id, peer_id, role, content)

    async def get_note(self, note_id: str) -> Note | None:
        """Get a note by ID."""
        return await self._sqlite.get_note(note_id)

    # ── Core coordination ──────────────────────────────────────────

    async def add_note(
        self,
        peer_id: str,
        content: str,
        session_id: str | None = None,
        note_type: str = "observation",
        provenance: str = "organic",
        durability: str = "contextual",
        emotional_weight: float = 0.5,
        keywords: list[str] | None = None,
        tags: list[str] | None = None,
        context_description: str | None = None,
    ) -> Note:
        """Embed content, store in both SQLite and Zvec, and return the Note."""
        # 1. Embed
        embedding = await asyncio.to_thread(self._embedder.embed_document, content)

        # 2. Insert into SQLite (zvec_id not yet known)
        note = await self._sqlite.create_note(
            peer_id=peer_id,
            content=content,
            session_id=session_id,
            context_description=context_description,
            keywords=keywords,
            tags=tags,
            note_type=note_type,
            provenance=provenance,
            durability=durability,
            emotional_weight=emotional_weight,
            zvec_id=None,
        )

        # 3. Insert into Zvec using the note's SQLite ID
        await asyncio.to_thread(self._zvec.insert, note.id, embedding)

        # 4. Back-fill zvec_id in SQLite
        note = await self._sqlite.update_note(note.id, zvec_id=note.id)
        logger.info("Added note %s for peer %s", note.id, peer_id)
        return note

    # ── Retrieval (Phase 3) ──────────────────────────────────────

    async def retrieve(
        self, query: str, peer_id: str, limit: int = 10
    ) -> list[RetrievalResult]:
        """Run the full retrieval pipeline with scoring and dedup."""
        return await self._retriever.retrieve(query, peer_id, limit=limit)

    # ── Search ─────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Strip characters that are special in FTS5 query syntax."""
        # Keep only word characters and whitespace, then collapse spaces
        cleaned = re.sub(r"[^\w\s]", " ", query)
        tokens = cleaned.split()
        return " ".join(tokens) if tokens else ""

    async def search_keyword(
        self, query: str, peer_id: str, limit: int = 20
    ) -> list[Note]:
        """Full-text keyword search, returning notes without scores."""
        sanitized = self._sanitize_fts_query(query)
        if not sanitized:
            return []
        results = await self._sqlite.fts_search(sanitized, peer_id, limit)
        return [note for note, _ in results]

    async def search_vector(
        self, query: str, peer_id: str, top_k: int = 20
    ) -> list[tuple[Note, float]]:
        """Semantic vector search filtered by peer."""
        query_embedding = await asyncio.to_thread(self._embedder.embed_query, query)
        vec_results = await asyncio.to_thread(self._zvec.search, query_embedding, top_k)

        results: list[tuple[Note, float]] = []
        for hit in vec_results:
            note = await self._sqlite.get_note(hit["id"])
            if note is not None and note.peer_id == peer_id:
                results.append((note, hit["score"]))
        return results

    async def search_hybrid(
        self, query: str, peer_id: str, limit: int = 10
    ) -> list[tuple[Note, float]]:
        """Hybrid search combining keyword + vector via Reciprocal Rank Fusion."""
        candidates = limit * 3

        kw_results, vec_results = await asyncio.gather(
            self.search_keyword(query, peer_id, limit=candidates),
            self.search_vector(query, peer_id, top_k=candidates),
        )

        rrf_scores: dict[str, float] = {}
        note_map: dict[str, Note] = {}

        for rank, note in enumerate(kw_results, start=1):
            rrf_scores[note.id] = rrf_scores.get(note.id, 0.0) + 1.0 / (RRF_K + rank)
            note_map[note.id] = note

        for rank, (note, _score) in enumerate(vec_results, start=1):
            rrf_scores[note.id] = rrf_scores.get(note.id, 0.0) + 1.0 / (RRF_K + rank)
            note_map[note.id] = note

        sorted_ids = sorted(rrf_scores, key=lambda nid: rrf_scores[nid], reverse=True)
        return [(note_map[nid], rrf_scores[nid]) for nid in sorted_ids[:limit]]
