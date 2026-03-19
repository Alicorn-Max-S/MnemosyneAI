"""Tests for the Intake fast-path ingestion."""

import pytest

from mnemosyne.pipeline.intake import ingest_message


class TestIntake:
    async def test_user_message_writes_and_enqueues(self, store):
        peer = await store.create_peer("Alice")
        session = await store.create_session(peer.id)

        msg_id = await ingest_message(session.id, peer.id, "user", "Hello", db=store)
        assert len(msg_id) == 26

        # A derive task should be enqueued
        task = await store.dequeue_task("derive")
        assert task is not None
        assert task.payload["message_id"] == msg_id
        assert task.payload["content"] == "Hello"

    async def test_assistant_message_no_enqueue(self, store):
        peer = await store.create_peer("Alice")
        session = await store.create_session(peer.id)

        msg_id = await ingest_message(session.id, peer.id, "assistant", "Hi there", db=store)
        assert len(msg_id) == 26

        # No derive task for assistant messages
        task = await store.dequeue_task("derive")
        assert task is None

    async def test_preceding_context_gathered(self, store):
        peer = await store.create_peer("Alice")
        session = await store.create_session(peer.id)

        # Add 5 prior messages
        for i in range(5):
            role = "user" if i % 2 == 0 else "assistant"
            await store.add_message(session.id, peer.id, role, f"msg {i}")

        # Ingest the 6th message
        msg_id = await ingest_message(session.id, peer.id, "user", "msg 5", db=store)

        task = await store.dequeue_task("derive")
        preceding = task.payload["preceding_turns"]
        # get_recent_context(n_turns=3) returns 3 most recent, minus the new message = 2
        assert len(preceding) == 2
        # The new message itself should not be in preceding
        assert all(t["id"] != msg_id for t in preceding)

    async def test_message_retrievable_after_ingestion(self, store):
        peer = await store.create_peer("Alice")
        session = await store.create_session(peer.id)

        msg_id = await ingest_message(session.id, peer.id, "user", "findme", db=store)

        messages = await store.get_messages(session.id)
        assert any(m.id == msg_id and m.content == "findme" for m in messages)

    async def test_payload_structure(self, store):
        peer = await store.create_peer("Alice")
        session = await store.create_session(peer.id)

        await ingest_message(session.id, peer.id, "user", "test content", db=store)
        task = await store.dequeue_task("derive")

        required_keys = {"message_id", "session_id", "peer_id", "content", "preceding_turns"}
        assert required_keys == set(task.payload.keys())
        assert task.payload["session_id"] == session.id
        assert task.payload["peer_id"] == peer.id
        assert task.payload["content"] == "test content"
        assert isinstance(task.payload["preceding_turns"], list)
