"""Tests for SQLiteStore."""

import pytest
import aiosqlite


# ── Schema ──────────────────────────────────────────────────────────


class TestSchema:
    async def test_tables_created(self, store):
        cursor = await store._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in await cursor.fetchall()}
        expected = {"config", "peers", "sessions", "messages", "notes", "links", "task_queue", "notes_fts", "peer_profiles", "entity_mentions", "colbert_tokens"}
        assert expected.issubset(tables)

    async def test_indexes_created(self, store):
        cursor = await store._db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        )
        indexes = {row[0] for row in await cursor.fetchall()}
        assert "idx_notes_peer_id" in indexes
        assert "idx_task_queue_status_priority" in indexes

    async def test_schema_version(self, store):
        cursor = await store._db.execute(
            "SELECT value FROM config WHERE key = 'schema_version'"
        )
        row = await cursor.fetchone()
        assert row[0] == "1"


# ── Peers ───────────────────────────────────────────────────────────


class TestPeers:
    async def test_create_and_get(self, store):
        peer = await store.create_peer("Alice")
        assert peer.name == "Alice"
        assert peer.peer_type == "user"
        assert len(peer.id) == 26

        fetched = await store.get_peer(peer.id)
        assert fetched is not None
        assert fetched.name == "Alice"

    async def test_get_not_found(self, store):
        result = await store.get_peer("nonexistent")
        assert result is None

    async def test_update(self, store):
        peer = await store.create_peer("Bob")
        updated = await store.update_peer(peer.id, name="Robert", static_profile={"bio": "dev"})
        assert updated.name == "Robert"
        assert updated.static_profile == {"bio": "dev"}
        assert updated.profile_updated_at is not None

    async def test_list(self, store):
        await store.create_peer("Alice")
        await store.create_peer("Bob")
        peers = await store.list_peers()
        assert len(peers) == 2


# ── Sessions ────────────────────────────────────────────────────────


class TestSessions:
    async def test_create_and_get(self, store):
        peer = await store.create_peer("Alice")
        session = await store.create_session(peer.id)
        assert session.peer_id == peer.id
        assert session.ended_at is None

    async def test_end_session(self, store):
        peer = await store.create_peer("Alice")
        session = await store.create_session(peer.id)
        ended = await store.end_session(session.id, summary="Good chat")
        assert ended.ended_at is not None
        assert ended.summary == "Good chat"

    async def test_list_sessions(self, store):
        peer = await store.create_peer("Alice")
        await store.create_session(peer.id)
        await store.create_session(peer.id)
        sessions = await store.list_sessions(peer.id)
        assert len(sessions) == 2


# ── Messages ────────────────────────────────────────────────────────


class TestMessages:
    async def test_add_and_get(self, store):
        peer = await store.create_peer("Alice")
        session = await store.create_session(peer.id)
        msg = await store.add_message(session.id, peer.id, "user", "Hello")
        assert msg.content == "Hello"
        assert msg.role == "user"

        messages = await store.get_messages(session.id)
        assert len(messages) == 1

    async def test_get_order(self, store):
        peer = await store.create_peer("Alice")
        session = await store.create_session(peer.id)
        await store.add_message(session.id, peer.id, "user", "First")
        await store.add_message(session.id, peer.id, "assistant", "Second")
        await store.add_message(session.id, peer.id, "user", "Third")

        messages = await store.get_messages(session.id)
        assert [m.content for m in messages] == ["First", "Second", "Third"]

    async def test_get_recent_context(self, store):
        peer = await store.create_peer("Alice")
        session = await store.create_session(peer.id)
        await store.add_message(session.id, peer.id, "user", "One")
        await store.add_message(session.id, peer.id, "assistant", "Two")
        await store.add_message(session.id, peer.id, "user", "Three")

        recent = await store.get_recent_context(session.id, 2)
        assert len(recent) == 2
        assert recent[0].content == "Two"
        assert recent[1].content == "Three"


# ── Notes ───────────────────────────────────────────────────────────


class TestNotes:
    async def test_create_minimal(self, store):
        peer = await store.create_peer("Alice")
        note = await store.create_note(peer.id, "A simple note")
        assert note.content == "A simple note"
        assert note.is_buffered is True
        assert note.note_type == "observation"

    async def test_create_all_fields(self, store):
        peer = await store.create_peer("Alice")
        session = await store.create_session(peer.id)
        note = await store.create_note(
            peer_id=peer.id,
            content="Detailed note",
            session_id=session.id,
            context_description="During onboarding",
            keywords=["python", "ai"],
            tags=["important"],
            note_type="inference",
            provenance="agent_prompted",
            durability="permanent",
            emotional_weight=0.9,
            importance=0.8,
            confidence=0.7,
            is_buffered=False,
        )
        assert note.keywords == ["python", "ai"]
        assert note.tags == ["important"]
        assert note.note_type == "inference"
        assert note.is_buffered is False
        assert note.importance == 0.8

    async def test_get(self, store):
        peer = await store.create_peer("Alice")
        note = await store.create_note(peer.id, "Test note")
        fetched = await store.get_note(note.id)
        assert fetched is not None
        assert fetched.content == "Test note"

    async def test_get_not_found(self, store):
        result = await store.get_note("nonexistent")
        assert result is None

    async def test_update(self, store):
        peer = await store.create_peer("Alice")
        note = await store.create_note(peer.id, "Original")
        updated = await store.update_note(note.id, content="Updated", importance=0.9)
        assert updated.content == "Updated"
        assert updated.importance == 0.9
        assert updated.updated_at != note.updated_at

    async def test_delete(self, store):
        peer = await store.create_peer("Alice")
        note = await store.create_note(peer.id, "To delete")
        assert await store.delete_note(note.id) is True
        assert await store.get_note(note.id) is None

    async def test_delete_not_found(self, store):
        assert await store.delete_note("nonexistent") is False

    async def test_list_with_filters(self, store):
        peer = await store.create_peer("Alice")
        await store.create_note(peer.id, "Obs 1", note_type="observation")
        await store.create_note(peer.id, "Obs 2", note_type="observation")
        await store.create_note(peer.id, "Inf 1", note_type="inference")

        all_notes = await store.list_notes(peer.id)
        assert len(all_notes) == 3

        obs_notes = await store.list_notes(peer.id, note_type="observation")
        assert len(obs_notes) == 2

        limited = await store.list_notes(peer.id, limit=1)
        assert len(limited) == 1

    async def test_get_buffered(self, store):
        peer = await store.create_peer("Alice")
        await store.create_note(peer.id, "Buffered", is_buffered=True)
        await store.create_note(peer.id, "Not buffered", is_buffered=False)

        buffered = await store.get_buffered_notes(peer.id)
        assert len(buffered) == 1
        assert buffered[0].content == "Buffered"


# ── Links ───────────────────────────────────────────────────────────


class TestLinks:
    async def test_create_and_get(self, store):
        peer = await store.create_peer("Alice")
        n1 = await store.create_note(peer.id, "Note 1")
        n2 = await store.create_note(peer.id, "Note 2")
        link = await store.create_link(n1.id, n2.id, "semantic", strength=0.8)
        assert link.link_type == "semantic"
        assert link.strength == 0.8

    async def test_get_bidirectional(self, store):
        peer = await store.create_peer("Alice")
        n1 = await store.create_note(peer.id, "Note 1")
        n2 = await store.create_note(peer.id, "Note 2")
        await store.create_link(n1.id, n2.id, "semantic")

        # Should find when querying from either side
        links_from_n1 = await store.get_links(n1.id)
        links_from_n2 = await store.get_links(n2.id)
        assert len(links_from_n1) == 1
        assert len(links_from_n2) == 1

    async def test_delete(self, store):
        peer = await store.create_peer("Alice")
        n1 = await store.create_note(peer.id, "Note 1")
        n2 = await store.create_note(peer.id, "Note 2")
        link = await store.create_link(n1.id, n2.id, "semantic")
        assert await store.delete_link(link.id) is True
        assert await store.get_links(n1.id) == []

    async def test_unique_constraint(self, store):
        peer = await store.create_peer("Alice")
        n1 = await store.create_note(peer.id, "Note 1")
        n2 = await store.create_note(peer.id, "Note 2")
        await store.create_link(n1.id, n2.id, "semantic")
        with pytest.raises(Exception):
            await store.create_link(n1.id, n2.id, "semantic")


# ── FTS5 ────────────────────────────────────────────────────────────


class TestFTS5:
    async def test_basic_search(self, store):
        peer = await store.create_peer("Alice")
        await store.create_note(peer.id, "Python is a great programming language")
        results = await store.fts_search("python", peer.id)
        assert len(results) == 1
        note, score = results[0]
        assert "Python" in note.content
        assert score > 0

    async def test_peer_filter(self, store):
        p1 = await store.create_peer("Alice")
        p2 = await store.create_peer("Bob")
        await store.create_note(p1.id, "Python programming")
        await store.create_note(p2.id, "Python scripting")

        results = await store.fts_search("python", p1.id)
        assert len(results) == 1
        assert results[0][0].peer_id == p1.id

    async def test_trigger_insert_searchable(self, store):
        peer = await store.create_peer("Alice")
        await store.create_note(peer.id, "Unique quantum computing topic")
        results = await store.fts_search("quantum", peer.id)
        assert len(results) == 1

    async def test_trigger_delete_gone(self, store):
        peer = await store.create_peer("Alice")
        note = await store.create_note(peer.id, "Temporary quantum data")
        results = await store.fts_search("quantum", peer.id)
        assert len(results) == 1

        await store.delete_note(note.id)
        results = await store.fts_search("quantum", peer.id)
        assert len(results) == 0

    async def test_trigger_update_reflects(self, store):
        peer = await store.create_peer("Alice")
        note = await store.create_note(peer.id, "Original quantum content")
        await store.update_note(note.id, content="Updated blockchain content")

        # Old content gone
        results = await store.fts_search("quantum", peer.id)
        assert len(results) == 0

        # New content found
        results = await store.fts_search("blockchain", peer.id)
        assert len(results) == 1

    async def test_no_results(self, store):
        peer = await store.create_peer("Alice")
        await store.create_note(peer.id, "Something else entirely")
        results = await store.fts_search("nonexistentterm", peer.id)
        assert len(results) == 0


# ── Task Queue ──────────────────────────────────────────────────────


class TestTaskQueue:
    async def test_enqueue(self, store):
        task = await store.enqueue_task("embed", payload={"note_id": "n1"}, priority=5)
        assert task.task_type == "embed"
        assert task.payload == {"note_id": "n1"}
        assert task.status == "pending"
        assert task.priority == 5

    async def test_dequeue(self, store):
        await store.enqueue_task("embed", payload={"note_id": "n1"})
        task = await store.dequeue_task("embed")
        assert task is not None
        assert task.status == "processing"
        assert task.attempts == 1
        assert task.started_at is not None

    async def test_dequeue_empty(self, store):
        result = await store.dequeue_task("embed")
        assert result is None

    async def test_dequeue_priority_order(self, store):
        await store.enqueue_task("embed", payload={"id": "low"}, priority=1)
        await store.enqueue_task("embed", payload={"id": "high"}, priority=10)
        await store.enqueue_task("embed", payload={"id": "mid"}, priority=5)

        task = await store.dequeue_task("embed")
        assert task.payload == {"id": "high"}

    async def test_complete(self, store):
        await store.enqueue_task("embed")
        task = await store.dequeue_task("embed")
        completed = await store.complete_task(task.id)
        assert completed.status == "completed"
        assert completed.completed_at is not None

    async def test_fail_retryable(self, store):
        await store.enqueue_task("embed", max_attempts=3)
        task = await store.dequeue_task("embed")
        assert task.attempts == 1
        failed = await store.fail_task(task.id, "Something went wrong")
        assert failed.status == "pending"
        assert failed.error == "Something went wrong"
        assert failed.completed_at is None

    async def test_dead_letter(self, store):
        await store.enqueue_task("embed", max_attempts=2)
        # First attempt: dequeue (attempts=1) then fail -> retryable
        task = await store.dequeue_task("embed")
        assert task.attempts == 1
        result = await store.fail_task(task.id, "error 1")
        assert result.status == "pending"

        # Second attempt: dequeue (attempts=2) then fail -> dead_letter
        task = await store.dequeue_task("embed")
        assert task.attempts == 2
        result = await store.fail_task(task.id, "error 2")
        assert result.status == "dead_letter"
        assert result.error == "error 2"
        assert result.completed_at is not None


# ── Entity Mentions ────────────────────────────────────────────────


class TestEntityMentions:
    async def test_add_and_get(self, store):
        peer = await store.create_peer("Alice")
        note = await store.create_note(peer.id, "Met with Bob today")
        mention = await store.add_entity_mention(
            note.id, peer.id, "Bob", "person", "Met with Bob today"
        )
        assert mention["entity_name"] == "Bob"
        assert mention["entity_type"] == "person"
        assert mention["note_id"] == note.id

        mentions = await store.get_entity_mentions(peer.id, "Bob")
        assert len(mentions) == 1
        assert mentions[0]["mention_context"] == "Met with Bob today"

    async def test_get_entities_for_peer(self, store):
        peer = await store.create_peer("Alice")
        n1 = await store.create_note(peer.id, "Bob is a developer")
        n2 = await store.create_note(peer.id, "Acme Corp is great")
        await store.add_entity_mention(n1.id, peer.id, "Bob", "person")
        await store.add_entity_mention(n2.id, peer.id, "Acme Corp", "organization")

        entities = await store.get_entities_for_peer(peer.id)
        assert len(entities) == 2
        names = {e["entity_name"] for e in entities}
        assert names == {"Bob", "Acme Corp"}

    async def test_peer_isolation(self, store):
        p1 = await store.create_peer("Alice")
        p2 = await store.create_peer("Bob")
        n1 = await store.create_note(p1.id, "Entity X")
        n2 = await store.create_note(p2.id, "Entity X")
        await store.add_entity_mention(n1.id, p1.id, "Entity X", "other")
        await store.add_entity_mention(n2.id, p2.id, "Entity X", "other")

        mentions_p1 = await store.get_entity_mentions(p1.id, "Entity X")
        mentions_p2 = await store.get_entity_mentions(p2.id, "Entity X")
        assert len(mentions_p1) == 1
        assert len(mentions_p2) == 1
        assert mentions_p1[0]["peer_id"] == p1.id
        assert mentions_p2[0]["peer_id"] == p2.id


# ── ColBERT Tokens ─────────────────────────────────────────────────


class TestColbertTokens:
    async def test_store_and_get_roundtrip(self, store):
        peer = await store.create_peer("Alice")
        note = await store.create_note(peer.id, "Test note")
        blob = b"\x00\x01\x02\x03" * 96
        await store.store_colbert_tokens(note.id, blob, 4)

        result = await store.get_colbert_tokens([note.id])
        assert note.id in result
        assert result[note.id] == blob

    async def test_missing_ids_omitted(self, store):
        result = await store.get_colbert_tokens(["nonexistent_id"])
        assert result == {}

    async def test_empty_list(self, store):
        result = await store.get_colbert_tokens([])
        assert result == {}

    async def test_replace_on_duplicate(self, store):
        peer = await store.create_peer("Alice")
        note = await store.create_note(peer.id, "Test note")
        blob1 = b"\x00" * 384
        blob2 = b"\xff" * 384
        await store.store_colbert_tokens(note.id, blob1, 4)
        await store.store_colbert_tokens(note.id, blob2, 8)

        result = await store.get_colbert_tokens([note.id])
        assert result[note.id] == blob2

    async def test_batch_get_multiple(self, store):
        peer = await store.create_peer("Alice")
        n1 = await store.create_note(peer.id, "Note 1")
        n2 = await store.create_note(peer.id, "Note 2")
        blob1 = b"\x01" * 100
        blob2 = b"\x02" * 200
        await store.store_colbert_tokens(n1.id, blob1, 2)
        await store.store_colbert_tokens(n2.id, blob2, 3)

        result = await store.get_colbert_tokens([n1.id, n2.id])
        assert len(result) == 2
        assert result[n1.id] == blob1
        assert result[n2.id] == blob2


# ── Merge Notes ────────────────────────────────────────────────────


class TestMergeNotes:
    async def test_canonical_note_id_set_on_merged(self, store):
        peer = await store.create_peer("Alice")
        canonical = await store.create_note(peer.id, "Canonical", importance=0.9)
        merged1 = await store.create_note(peer.id, "Dup 1", importance=0.3)
        merged2 = await store.create_note(peer.id, "Dup 2", importance=0.2)

        await store.merge_notes(canonical.id, [merged1.id, merged2.id])

        m1 = await store.get_note(merged1.id)
        m2 = await store.get_note(merged2.id)
        assert m1.canonical_note_id == canonical.id
        assert m2.canonical_note_id == canonical.id

    async def test_is_buffered_cleared_on_all(self, store):
        peer = await store.create_peer("Alice")
        canonical = await store.create_note(peer.id, "Canonical", is_buffered=True)
        merged = await store.create_note(peer.id, "Dup", is_buffered=True)

        await store.merge_notes(canonical.id, [merged.id])

        c = await store.get_note(canonical.id)
        m = await store.get_note(merged.id)
        assert c.is_buffered is False
        assert m.is_buffered is False

    async def test_evidence_count_summed(self, store):
        peer = await store.create_peer("Alice")
        canonical = await store.create_note(peer.id, "Canonical")  # evidence_count=1
        merged1 = await store.create_note(peer.id, "Dup 1")  # evidence_count=1
        merged2 = await store.create_note(peer.id, "Dup 2")  # evidence_count=1

        await store.merge_notes(canonical.id, [merged1.id, merged2.id])

        c = await store.get_note(canonical.id)
        assert c.evidence_count == 3  # 1 (original) + 1 + 1 (merged)

    async def test_canonical_has_no_canonical_note_id(self, store):
        peer = await store.create_peer("Alice")
        canonical = await store.create_note(peer.id, "Canonical")
        merged = await store.create_note(peer.id, "Dup")

        await store.merge_notes(canonical.id, [merged.id])

        c = await store.get_note(canonical.id)
        assert c.canonical_note_id is None


# ── Get Unique Sessions ───────────────────────────────────────────


class TestGetUniqueSessions:
    async def test_counts_distinct_sessions(self, store):
        peer = await store.create_peer("Alice")
        s1 = await store.create_session(peer.id)
        s2 = await store.create_session(peer.id)
        n1 = await store.create_note(peer.id, "Note 1", session_id=s1.id)
        n2 = await store.create_note(peer.id, "Note 2", session_id=s1.id)
        n3 = await store.create_note(peer.id, "Note 3", session_id=s2.id)

        count = await store.get_unique_sessions_for_notes([n1.id, n2.id, n3.id])
        assert count == 2

    async def test_null_sessions_excluded(self, store):
        peer = await store.create_peer("Alice")
        s1 = await store.create_session(peer.id)
        n1 = await store.create_note(peer.id, "With session", session_id=s1.id)
        n2 = await store.create_note(peer.id, "No session")  # session_id=None

        count = await store.get_unique_sessions_for_notes([n1.id, n2.id])
        assert count == 1

    async def test_empty_list_returns_zero(self, store):
        count = await store.get_unique_sessions_for_notes([])
        assert count == 0
