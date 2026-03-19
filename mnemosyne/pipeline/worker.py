"""Background worker: polls task queue and dispatches to handlers."""

import asyncio
import logging
from collections.abc import Callable

from mnemosyne.config import WORKER_POLL_INTERVAL
from mnemosyne.db.sqlite_store import SQLiteStore
from mnemosyne.models import TaskItem

logger = logging.getLogger(__name__)

HandlerFn = Callable[[TaskItem], object]  # async (TaskItem) -> None


class Worker:
    """Polls the task queue and dispatches tasks to registered handlers."""

    def __init__(self, db: SQLiteStore, handler_map: dict[str, HandlerFn]) -> None:
        """Initialize with a SQLiteStore and a map of task_type -> async handler."""
        self.db = db
        self.handler_map = handler_map
        self._running = False

    async def run(self, poll_interval: float = WORKER_POLL_INTERVAL) -> None:
        """Poll for tasks in a loop until stopped."""
        self._running = True
        logger.info("Worker started, polling every %.1fs", poll_interval)
        while self._running:
            processed = await self.run_once()
            if not processed:
                await asyncio.sleep(poll_interval)

    def stop(self) -> None:
        """Signal the worker to stop after the current iteration."""
        self._running = False

    async def run_once(self) -> bool:
        """Attempt to dequeue and process one task. Returns True if a task was processed."""
        for task_type in self.handler_map:
            task = await self.db.dequeue_task(task_type)
            if task is not None:
                await self._dispatch(task)
                return True
        return False

    async def _dispatch(self, task: TaskItem) -> None:
        """Look up handler for task type, call it, and complete or fail the task."""
        handler = self.handler_map.get(task.task_type)
        if handler is None:
            logger.error("No handler for task type %r, failing task %s", task.task_type, task.id)
            await self.db.fail_task(task.id, f"Unknown task type: {task.task_type}")
            return

        try:
            await handler(task)
            await self.db.complete_task(task.id)
            logger.info("Task %s (%s) completed", task.id, task.task_type)
        except Exception as exc:
            logger.warning("Task %s (%s) failed: %s", task.id, task.task_type, exc)
            await self.db.fail_task(task.id, str(exc))
