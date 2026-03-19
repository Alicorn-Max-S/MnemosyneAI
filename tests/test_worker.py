"""Tests for the Worker queue poller."""

import pytest

from mnemosyne.pipeline.worker import Worker


class TestWorker:
    async def test_processes_pending_task(self, store):
        called_with = []

        async def handler(task):
            called_with.append(task)

        await store.enqueue_task("embed", payload={"note_id": "n1"})
        worker = Worker(store, {"embed": handler})
        result = await worker.run_once()

        assert result is True
        assert len(called_with) == 1
        assert called_with[0].payload == {"note_id": "n1"}

    async def test_returns_false_when_empty(self, store):
        async def handler(task):
            pass

        worker = Worker(store, {"embed": handler})
        result = await worker.run_once()
        assert result is False

    async def test_completes_task_on_success(self, store):
        async def handler(task):
            pass

        await store.enqueue_task("embed", payload={"note_id": "n1"})
        worker = Worker(store, {"embed": handler})
        await worker.run_once()

        # The task should now be completed — dequeue returns None
        task = await store.dequeue_task("embed")
        assert task is None

    async def test_catches_handler_exception(self, store):
        async def bad_handler(task):
            raise ValueError("oops")

        await store.enqueue_task("embed", max_attempts=3)
        worker = Worker(store, {"embed": bad_handler})
        result = await worker.run_once()

        assert result is True
        # Task should be back to pending (retryable) since attempts=1 < max_attempts=3
        task = await store.dequeue_task("embed")
        assert task is not None
        assert task.status == "processing"
        assert task.attempts == 2

    async def test_dead_letter_after_max_attempts(self, store):
        async def bad_handler(task):
            raise RuntimeError("still broken")

        await store.enqueue_task("embed", max_attempts=2)
        worker = Worker(store, {"embed": bad_handler})

        # First attempt: dequeue (attempts=1), handler fails, fail_task -> pending
        await worker.run_once()
        # Second attempt: dequeue (attempts=2), handler fails, fail_task -> dead_letter
        await worker.run_once()

        # No more tasks to dequeue
        task = await store.dequeue_task("embed")
        assert task is None

    async def test_atomic_dequeue_prevents_double_processing(self, store):
        call_count = 0

        async def handler(task):
            nonlocal call_count
            call_count += 1

        await store.enqueue_task("embed")
        worker = Worker(store, {"embed": handler})

        r1 = await worker.run_once()
        r2 = await worker.run_once()

        assert r1 is True
        assert r2 is False
        assert call_count == 1
