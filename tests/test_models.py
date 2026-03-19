"""Tests for Pydantic data models."""

from mnemosyne.models import Link, Message, Note, Peer, Session, TaskItem


class TestPeer:
    def test_create_with_defaults(self):
        p = Peer(id="abc", name="Alice", created_at="2024-01-01T00:00:00Z")
        assert p.peer_type == "user"
        assert p.static_profile is None
        assert p.metadata == {}

    def test_from_row(self):
        row = {
            "id": "abc",
            "name": "Alice",
            "peer_type": "agent",
            "static_profile": '{"role": "helper"}',
            "profile_updated_at": None,
            "created_at": "2024-01-01T00:00:00Z",
            "metadata": '{"key": "val"}',
        }
        p = Peer.from_row(row)
        assert p.static_profile == {"role": "helper"}
        assert p.metadata == {"key": "val"}

    def test_from_row_null_profile(self):
        row = {
            "id": "abc",
            "name": "Bob",
            "peer_type": "user",
            "static_profile": None,
            "profile_updated_at": None,
            "created_at": "2024-01-01T00:00:00Z",
            "metadata": "{}",
        }
        p = Peer.from_row(row)
        assert p.static_profile is None


class TestSession:
    def test_create_with_defaults(self):
        s = Session(id="s1", peer_id="p1", started_at="2024-01-01T00:00:00Z")
        assert s.ended_at is None
        assert s.summary is None
        assert s.metadata == {}

    def test_from_row(self):
        row = {
            "id": "s1",
            "peer_id": "p1",
            "started_at": "2024-01-01T00:00:00Z",
            "ended_at": None,
            "summary": None,
            "metadata": '{"channel": "web"}',
        }
        s = Session.from_row(row)
        assert s.metadata == {"channel": "web"}


class TestMessage:
    def test_create(self):
        m = Message(
            id="m1", session_id="s1", peer_id="p1",
            role="user", content="hello", created_at="2024-01-01T00:00:00Z",
        )
        assert m.metadata == {}

    def test_from_row(self):
        row = {
            "id": "m1",
            "session_id": "s1",
            "peer_id": "p1",
            "role": "assistant",
            "content": "hi",
            "created_at": "2024-01-01T00:00:00Z",
            "metadata": '{"tokens": 5}',
        }
        m = Message.from_row(row)
        assert m.metadata == {"tokens": 5}


class TestNote:
    def test_create_with_defaults(self):
        n = Note(
            id="n1", peer_id="p1", content="test",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
        assert n.keywords == []
        assert n.tags == []
        assert n.is_buffered is True
        assert n.note_type == "observation"
        assert n.confidence == 0.8

    def test_from_row_json_and_bool(self):
        row = {
            "id": "n1",
            "peer_id": "p1",
            "session_id": None,
            "source_message_id": None,
            "content": "test",
            "context_description": None,
            "keywords": '["python", "ai"]',
            "tags": '["important"]',
            "note_type": "observation",
            "provenance": "organic",
            "durability": "contextual",
            "emotional_weight": 0.5,
            "importance": 0.0,
            "confidence": 0.8,
            "evidence_count": 1,
            "unique_sessions_mentioned": 1,
            "q_value": 0.0,
            "access_count": 0,
            "last_accessed_at": None,
            "times_surfaced": 0,
            "decay_score": 1.0,
            "is_buffered": 1,
            "canonical_note_id": None,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "zvec_id": None,
        }
        n = Note.from_row(row)
        assert n.keywords == ["python", "ai"]
        assert n.tags == ["important"]
        assert n.is_buffered is True

    def test_is_buffered_false(self):
        row = {
            "id": "n2",
            "peer_id": "p1",
            "session_id": None,
            "source_message_id": None,
            "content": "test",
            "context_description": None,
            "keywords": "[]",
            "tags": "[]",
            "note_type": "observation",
            "provenance": "organic",
            "durability": "contextual",
            "emotional_weight": 0.5,
            "importance": 0.0,
            "confidence": 0.8,
            "evidence_count": 1,
            "unique_sessions_mentioned": 1,
            "q_value": 0.0,
            "access_count": 0,
            "last_accessed_at": None,
            "times_surfaced": 0,
            "decay_score": 1.0,
            "is_buffered": 0,
            "canonical_note_id": None,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "zvec_id": None,
        }
        n = Note.from_row(row)
        assert n.is_buffered is False


class TestLink:
    def test_create_with_defaults(self):
        link = Link(
            id="l1", source_note_id="n1", target_note_id="n2",
            link_type="semantic", created_at="2024-01-01T00:00:00Z",
        )
        assert link.strength == 0.5
        assert link.metadata == {}

    def test_from_row(self):
        row = {
            "id": "l1",
            "source_note_id": "n1",
            "target_note_id": "n2",
            "link_type": "causal",
            "strength": 0.9,
            "created_at": "2024-01-01T00:00:00Z",
            "metadata": '{"reason": "test"}',
        }
        link = Link.from_row(row)
        assert link.metadata == {"reason": "test"}


class TestTaskItem:
    def test_create_with_defaults(self):
        t = TaskItem(
            id="t1", task_type="embed",
            created_at="2024-01-01T00:00:00Z",
        )
        assert t.payload == {}
        assert t.status == "pending"
        assert t.priority == 0
        assert t.max_attempts == 3

    def test_from_row(self):
        row = {
            "id": "t1",
            "task_type": "embed",
            "payload": '{"note_id": "n1"}',
            "status": "processing",
            "priority": 5,
            "attempts": 1,
            "max_attempts": 3,
            "error": None,
            "created_at": "2024-01-01T00:00:00Z",
            "started_at": "2024-01-01T00:00:01Z",
            "completed_at": None,
        }
        t = TaskItem.from_row(row)
        assert t.payload == {"note_id": "n1"}
        assert t.status == "processing"
