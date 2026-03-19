"""SQLite store with FTS5 for Mnemosyne."""

import json
import logging

import aiosqlite

from mnemosyne.models import Link, Message, Note, Peer, Session, TaskItem
from mnemosyne.utils.ids import generate_id

logger = logging.getLogger(__name__)


class SQLiteStore:
    """Async SQLite store with FTS5 full-text search."""

    def __init__(self, db_path: str) -> None:
        """Initialize with database file path."""
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Connect to database, apply PRAGMAs, create schema."""
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row

        # PRAGMAs
        await self._db.execute("PRAGMA journal_mode = WAL")
        await self._db.execute("PRAGMA synchronous = NORMAL")
        await self._db.execute("PRAGMA cache_size = -8000")
        await self._db.execute("PRAGMA busy_timeout = 5000")
        await self._db.execute("PRAGMA mmap_size = 268435456")
        await self._db.execute("PRAGMA temp_store = MEMORY")
        await self._db.execute("PRAGMA foreign_keys = ON")

        await self._create_tables()
        await self._create_fts()
        await self._create_indexes()

        # Insert schema version
        await self._db.execute(
            """INSERT OR IGNORE INTO config (key, value, updated_at)
               VALUES ('schema_version', '1', strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"""
        )
        await self._db.commit()
        logger.info("SQLiteStore initialized at %s", self.db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    # ── Schema ──────────────────────────────────────────────────────

    async def _create_tables(self) -> None:
        """Create all tables."""
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            );

            CREATE TABLE IF NOT EXISTS peers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                peer_type TEXT NOT NULL DEFAULT 'user',
                static_profile TEXT,
                profile_updated_at TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                peer_id TEXT NOT NULL REFERENCES peers(id),
                started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                ended_at TEXT,
                summary TEXT,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                peer_id TEXT NOT NULL REFERENCES peers(id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                peer_id TEXT NOT NULL REFERENCES peers(id),
                session_id TEXT REFERENCES sessions(id),
                source_message_id TEXT REFERENCES messages(id),
                content TEXT NOT NULL,
                context_description TEXT,
                keywords TEXT NOT NULL DEFAULT '[]',
                tags TEXT NOT NULL DEFAULT '[]',
                note_type TEXT NOT NULL DEFAULT 'observation',
                provenance TEXT NOT NULL DEFAULT 'organic',
                durability TEXT NOT NULL DEFAULT 'contextual',
                emotional_weight REAL NOT NULL DEFAULT 0.5,
                importance REAL NOT NULL DEFAULT 0.0,
                confidence REAL NOT NULL DEFAULT 0.8,
                evidence_count INTEGER NOT NULL DEFAULT 1,
                unique_sessions_mentioned INTEGER NOT NULL DEFAULT 1,
                q_value REAL NOT NULL DEFAULT 0.0,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed_at TEXT,
                times_surfaced INTEGER NOT NULL DEFAULT 0,
                decay_score REAL NOT NULL DEFAULT 1.0,
                is_buffered INTEGER NOT NULL DEFAULT 1,
                canonical_note_id TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                zvec_id TEXT
            );

            CREATE TABLE IF NOT EXISTS links (
                id TEXT PRIMARY KEY,
                source_note_id TEXT NOT NULL REFERENCES notes(id),
                target_note_id TEXT NOT NULL REFERENCES notes(id),
                link_type TEXT NOT NULL,
                strength REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                metadata TEXT NOT NULL DEFAULT '{}',
                UNIQUE(source_note_id, target_note_id, link_type)
            );

            CREATE TABLE IF NOT EXISTS task_queue (
                id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                payload TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                priority INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3,
                error TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                started_at TEXT,
                completed_at TEXT
            );
        """)

    async def _create_fts(self) -> None:
        """Create FTS5 virtual table and sync triggers."""
        await self._db.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                content, context_description, keywords, tags,
                content='notes', content_rowid='rowid'
            );

            CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
                INSERT INTO notes_fts(rowid, content, context_description, keywords, tags)
                VALUES (new.rowid, new.content, new.context_description, new.keywords, new.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
                INSERT INTO notes_fts(notes_fts, rowid, content, context_description, keywords, tags)
                VALUES ('delete', old.rowid, old.content, old.context_description, old.keywords, old.tags);
            END;

            CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
                INSERT INTO notes_fts(notes_fts, rowid, content, context_description, keywords, tags)
                VALUES ('delete', old.rowid, old.content, old.context_description, old.keywords, old.tags);
                INSERT INTO notes_fts(rowid, content, context_description, keywords, tags)
                VALUES (new.rowid, new.content, new.context_description, new.keywords, new.tags);
            END;
        """)

    async def _create_indexes(self) -> None:
        """Create all indexes."""
        await self._db.executescript("""
            CREATE INDEX IF NOT EXISTS idx_sessions_peer_id ON sessions(peer_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at);
            CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
            CREATE INDEX IF NOT EXISTS idx_notes_peer_id ON notes(peer_id);
            CREATE INDEX IF NOT EXISTS idx_notes_session_id ON notes(session_id);
            CREATE INDEX IF NOT EXISTS idx_notes_note_type ON notes(note_type);
            CREATE INDEX IF NOT EXISTS idx_notes_durability ON notes(durability);
            CREATE INDEX IF NOT EXISTS idx_notes_is_buffered ON notes(is_buffered);
            CREATE INDEX IF NOT EXISTS idx_notes_decay_score ON notes(decay_score);
            CREATE INDEX IF NOT EXISTS idx_notes_importance ON notes(importance);
            CREATE INDEX IF NOT EXISTS idx_notes_created_at ON notes(created_at);
            CREATE INDEX IF NOT EXISTS idx_links_source ON links(source_note_id);
            CREATE INDEX IF NOT EXISTS idx_links_target ON links(target_note_id);
            CREATE INDEX IF NOT EXISTS idx_links_type ON links(link_type);
            CREATE INDEX IF NOT EXISTS idx_task_queue_status_priority ON task_queue(status, priority DESC);
        """)

    # ── Peers ───────────────────────────────────────────────────────

    async def create_peer(
        self,
        name: str,
        peer_type: str = "user",
        static_profile: dict | None = None,
        metadata: dict | None = None,
    ) -> Peer:
        """Create a new peer."""
        peer_id = generate_id()
        profile_json = json.dumps(static_profile) if static_profile else None
        meta_json = json.dumps(metadata or {})
        await self._db.execute(
            """INSERT INTO peers (id, name, peer_type, static_profile, metadata)
               VALUES (?, ?, ?, ?, ?)""",
            (peer_id, name, peer_type, profile_json, meta_json),
        )
        await self._db.commit()
        return await self.get_peer(peer_id)

    async def get_peer(self, peer_id: str) -> Peer | None:
        """Get a peer by ID."""
        cursor = await self._db.execute(
            "SELECT * FROM peers WHERE id = ?", (peer_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return Peer.from_row(dict(row))

    async def update_peer(self, peer_id: str, **kwargs: object) -> Peer | None:
        """Update peer fields dynamically."""
        if not kwargs:
            return await self.get_peer(peer_id)

        sets = []
        values = []
        for key, value in kwargs.items():
            if key in ("static_profile", "metadata") and isinstance(value, dict):
                value = json.dumps(value)
            sets.append(f"{key} = ?")
            values.append(value)

        if "static_profile" in kwargs:
            sets.append("profile_updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')")

        values.append(peer_id)
        sql = f"UPDATE peers SET {', '.join(sets)} WHERE id = ?"
        await self._db.execute(sql, values)
        await self._db.commit()
        return await self.get_peer(peer_id)

    async def list_peers(self) -> list[Peer]:
        """List all peers."""
        cursor = await self._db.execute("SELECT * FROM peers ORDER BY created_at")
        rows = await cursor.fetchall()
        return [Peer.from_row(dict(r)) for r in rows]

    # ── Sessions ────────────────────────────────────────────────────

    async def create_session(
        self, peer_id: str, metadata: dict | None = None
    ) -> Session:
        """Create a new session."""
        session_id = generate_id()
        meta_json = json.dumps(metadata or {})
        await self._db.execute(
            """INSERT INTO sessions (id, peer_id, metadata)
               VALUES (?, ?, ?)""",
            (session_id, peer_id, meta_json),
        )
        await self._db.commit()
        return await self.get_session(session_id)

    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        cursor = await self._db.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return Session.from_row(dict(row))

    async def end_session(
        self, session_id: str, summary: str | None = None
    ) -> Session | None:
        """End a session by setting ended_at."""
        await self._db.execute(
            """UPDATE sessions
               SET ended_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), summary = ?
               WHERE id = ?""",
            (summary, session_id),
        )
        await self._db.commit()
        return await self.get_session(session_id)

    async def list_sessions(self, peer_id: str) -> list[Session]:
        """List sessions for a peer."""
        cursor = await self._db.execute(
            "SELECT * FROM sessions WHERE peer_id = ? ORDER BY started_at",
            (peer_id,),
        )
        rows = await cursor.fetchall()
        return [Session.from_row(dict(r)) for r in rows]

    # ── Messages ────────────────────────────────────────────────────

    async def add_message(
        self,
        session_id: str,
        peer_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> Message:
        """Add a message to a session."""
        msg_id = generate_id()
        meta_json = json.dumps(metadata or {})
        await self._db.execute(
            """INSERT INTO messages (id, session_id, peer_id, role, content, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (msg_id, session_id, peer_id, role, content, meta_json),
        )
        await self._db.commit()
        cursor = await self._db.execute(
            "SELECT * FROM messages WHERE id = ?", (msg_id,)
        )
        row = await cursor.fetchone()
        return Message.from_row(dict(row))

    async def get_messages(
        self, session_id: str, limit: int | None = None
    ) -> list[Message]:
        """Get messages for a session, ordered by created_at."""
        sql = "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at"
        params: list[object] = [session_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [Message.from_row(dict(r)) for r in rows]

    async def get_recent_context(
        self, session_id: str, n_turns: int
    ) -> list[Message]:
        """Get the most recent n messages from a session."""
        cursor = await self._db.execute(
            """SELECT * FROM messages
               WHERE session_id = ? AND rowid IN (
                   SELECT rowid FROM messages WHERE session_id = ?
                   ORDER BY rowid DESC LIMIT ?
               )
               ORDER BY rowid""",
            (session_id, session_id, n_turns),
        )
        rows = await cursor.fetchall()
        return [Message.from_row(dict(r)) for r in rows]

    # ── Notes ───────────────────────────────────────────────────────

    async def create_note(
        self,
        peer_id: str,
        content: str,
        session_id: str | None = None,
        source_message_id: str | None = None,
        context_description: str | None = None,
        keywords: list[str] | None = None,
        tags: list[str] | None = None,
        note_type: str = "observation",
        provenance: str = "organic",
        durability: str = "contextual",
        emotional_weight: float = 0.5,
        importance: float = 0.0,
        confidence: float = 0.8,
        is_buffered: bool = True,
        zvec_id: str | None = None,
    ) -> Note:
        """Create a new note."""
        note_id = generate_id()
        await self._db.execute(
            """INSERT INTO notes (
                   id, peer_id, session_id, source_message_id, content,
                   context_description, keywords, tags, note_type, provenance,
                   durability, emotional_weight, importance, confidence,
                   is_buffered, zvec_id
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                note_id, peer_id, session_id, source_message_id, content,
                context_description,
                json.dumps(keywords or []),
                json.dumps(tags or []),
                note_type, provenance, durability,
                emotional_weight, importance, confidence,
                int(is_buffered), zvec_id,
            ),
        )
        await self._db.commit()
        return await self.get_note(note_id)

    async def get_note(self, note_id: str) -> Note | None:
        """Get a note by ID."""
        cursor = await self._db.execute(
            "SELECT * FROM notes WHERE id = ?", (note_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return Note.from_row(dict(row))

    async def update_note(self, note_id: str, **kwargs: object) -> Note | None:
        """Update note fields dynamically."""
        if not kwargs:
            return await self.get_note(note_id)

        sets = []
        values = []
        for key, value in kwargs.items():
            if key in ("keywords", "tags") and isinstance(value, list):
                value = json.dumps(value)
            elif key == "is_buffered" and isinstance(value, bool):
                value = int(value)
            sets.append(f"{key} = ?")
            values.append(value)

        sets.append("updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')")
        values.append(note_id)
        sql = f"UPDATE notes SET {', '.join(sets)} WHERE id = ?"
        await self._db.execute(sql, values)
        await self._db.commit()
        return await self.get_note(note_id)

    async def delete_note(self, note_id: str) -> bool:
        """Delete a note by ID."""
        cursor = await self._db.execute(
            "DELETE FROM notes WHERE id = ?", (note_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def list_notes(
        self,
        peer_id: str,
        note_type: str | None = None,
        durability: str | None = None,
        limit: int | None = None,
    ) -> list[Note]:
        """List notes for a peer with optional filters."""
        sql = "SELECT * FROM notes WHERE peer_id = ?"
        params: list[object] = [peer_id]
        if note_type is not None:
            sql += " AND note_type = ?"
            params.append(note_type)
        if durability is not None:
            sql += " AND durability = ?"
            params.append(durability)
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [Note.from_row(dict(r)) for r in rows]

    async def get_buffered_notes(self, peer_id: str) -> list[Note]:
        """Get all buffered notes for a peer."""
        cursor = await self._db.execute(
            "SELECT * FROM notes WHERE peer_id = ? AND is_buffered = 1 ORDER BY created_at",
            (peer_id,),
        )
        rows = await cursor.fetchall()
        return [Note.from_row(dict(r)) for r in rows]

    # ── Links ───────────────────────────────────────────────────────

    async def create_link(
        self,
        source_note_id: str,
        target_note_id: str,
        link_type: str,
        strength: float = 0.5,
        metadata: dict | None = None,
    ) -> Link:
        """Create a link between two notes."""
        link_id = generate_id()
        meta_json = json.dumps(metadata or {})
        await self._db.execute(
            """INSERT INTO links (id, source_note_id, target_note_id, link_type, strength, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (link_id, source_note_id, target_note_id, link_type, strength, meta_json),
        )
        await self._db.commit()
        cursor = await self._db.execute(
            "SELECT * FROM links WHERE id = ?", (link_id,)
        )
        row = await cursor.fetchone()
        return Link.from_row(dict(row))

    async def get_links(self, note_id: str) -> list[Link]:
        """Get all links involving a note (as source or target)."""
        cursor = await self._db.execute(
            """SELECT * FROM links
               WHERE source_note_id = ? OR target_note_id = ?
               ORDER BY created_at""",
            (note_id, note_id),
        )
        rows = await cursor.fetchall()
        return [Link.from_row(dict(r)) for r in rows]

    async def delete_link(self, link_id: str) -> bool:
        """Delete a link by ID."""
        cursor = await self._db.execute(
            "DELETE FROM links WHERE id = ?", (link_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0

    # ── Task Queue ──────────────────────────────────────────────────

    async def enqueue_task(
        self,
        task_type: str,
        payload: dict | None = None,
        priority: int = 0,
        max_attempts: int = 3,
    ) -> TaskItem:
        """Add a task to the queue."""
        task_id = generate_id()
        payload_json = json.dumps(payload or {})
        await self._db.execute(
            """INSERT INTO task_queue (id, task_type, payload, priority, max_attempts)
               VALUES (?, ?, ?, ?, ?)""",
            (task_id, task_type, payload_json, priority, max_attempts),
        )
        await self._db.commit()
        cursor = await self._db.execute(
            "SELECT * FROM task_queue WHERE id = ?", (task_id,)
        )
        row = await cursor.fetchone()
        return TaskItem.from_row(dict(row))

    async def dequeue_task(self, task_type: str) -> TaskItem | None:
        """Atomically claim the highest-priority pending task of the given type."""
        cursor = await self._db.execute(
            """UPDATE task_queue
               SET status = 'processing',
                   started_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                   attempts = attempts + 1
               WHERE id = (
                   SELECT id FROM task_queue
                   WHERE task_type = ? AND status = 'pending'
                   ORDER BY priority DESC, created_at ASC
                   LIMIT 1
               )
               RETURNING *""",
            (task_type,),
        )
        row = await cursor.fetchone()
        await self._db.commit()
        if row is None:
            return None
        return TaskItem.from_row(dict(row))

    async def complete_task(self, task_id: str) -> TaskItem | None:
        """Mark a task as completed."""
        await self._db.execute(
            """UPDATE task_queue
               SET status = 'completed',
                   completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
               WHERE id = ?""",
            (task_id,),
        )
        await self._db.commit()
        cursor = await self._db.execute(
            "SELECT * FROM task_queue WHERE id = ?", (task_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return TaskItem.from_row(dict(row))

    async def fail_task(self, task_id: str, error: str) -> TaskItem | None:
        """Fail a task: retry if attempts < max_attempts, otherwise dead_letter."""
        await self._db.execute(
            """UPDATE task_queue
               SET error = ?,
                   status = CASE
                       WHEN attempts >= max_attempts THEN 'dead_letter'
                       ELSE 'pending'
                   END,
                   started_at = CASE
                       WHEN attempts >= max_attempts THEN started_at
                       ELSE NULL
                   END,
                   completed_at = CASE
                       WHEN attempts >= max_attempts THEN strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                       ELSE NULL
                   END
               WHERE id = ?""",
            (error, task_id),
        )
        await self._db.commit()
        cursor = await self._db.execute(
            "SELECT * FROM task_queue WHERE id = ?", (task_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return TaskItem.from_row(dict(row))

    # ── FTS5 Search ─────────────────────────────────────────────────

    async def fts_search(
        self, query: str, peer_id: str, limit: int = 20
    ) -> list[tuple[Note, float]]:
        """Full-text search notes by peer, returning (Note, score) tuples."""
        cursor = await self._db.execute(
            """SELECT notes.*, -rank AS score
               FROM notes_fts
               JOIN notes ON notes.rowid = notes_fts.rowid
               WHERE notes_fts MATCH ? AND notes.peer_id = ?
               ORDER BY score DESC
               LIMIT ?""",
            (query, peer_id, limit),
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            row_dict = dict(row)
            score = row_dict.pop("score")
            results.append((Note.from_row(row_dict), score))
        return results
