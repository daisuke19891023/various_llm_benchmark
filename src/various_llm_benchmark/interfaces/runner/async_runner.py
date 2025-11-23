from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


@dataclass(slots=True)
class TaskResult[T]:
    """Result of an executed task."""

    name: str
    result: T | None = None
    error: BaseException | None = None
    attempts: int = 0
    cancelled: bool = False

    @property
    def succeeded(self) -> bool:
        """Return True when the task finished without errors."""
        return self.error is None and not self.cancelled


class TaskHooks:
    """Lifecycle callbacks for task execution."""

    def on_start(self, name: str, attempt: int) -> None:
        """Call when a task attempt is about to start."""

    def on_retry(self, name: str, attempt: int, error: BaseException) -> None:
        """Call when a task is re-enqueued for retry."""

    def on_success(self, name: str, attempt: int) -> None:
        """Call when a task completes successfully."""

    def on_failure(self, name: str, attempt: int, error: BaseException) -> None:
        """Call when a task gives up after retries."""

    def on_cancel(self, name: str, attempt: int) -> None:
        """Call when a task is cancelled."""


class AsyncJobRunner[T]:
    """Asyncio based job runner with retry and cancellation support."""

    def __init__(
        self,
        *,
        concurrency: int = 3,
        max_retries: int = 0,
        hooks: TaskHooks | None = None,
    ) -> None:
        """Initialize the runner with concurrency and retry settings."""
        if concurrency < 1:
            msg = "concurrency must be at least 1"
            raise ValueError(msg)
        if max_retries < 0:
            msg = "max_retries must be non-negative"
            raise ValueError(msg)

        self._concurrency = concurrency
        self._max_retries = max_retries
        self._tasks: list[tuple[str, Callable[[], Awaitable[T]]]] = []
        self._cancel_event = asyncio.Event()
        self._hooks = hooks or TaskHooks()

    def add_task(
        self,
        func: Callable[..., Awaitable[T]] | Callable[..., T],
        *args: Any,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Register a callable task with optional arguments."""

        async def _wrapped() -> T:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                return await cast("Awaitable[T]", result)
            return cast("T", result)

        task_name = name or getattr(func, "__name__", "task")
        self._tasks.append((task_name, _wrapped))

    async def run(self) -> list[TaskResult[T]]:
        """Execute all registered tasks respecting concurrency and retries."""
        queue: asyncio.Queue[tuple[int, str, Callable[[], Awaitable[T]], int]] = asyncio.Queue()
        results: list[TaskResult[T] | None] = [None for _ in self._tasks]

        for task_id, (task_name, task_callable) in enumerate(self._tasks):
            await queue.put((task_id, task_name, task_callable, 0))

        workers = [
            asyncio.create_task(self._worker(queue, results), name=f"async-runner-worker-{idx}")
            for idx in range(self._concurrency)
        ]

        try:
            await queue.join()
        finally:
            self._cancel_event.set()
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        return [result for result in results if result is not None]

    def request_cancel(self) -> None:
        """Signal cancellation for remaining tasks."""
        self._cancel_event.set()

    async def _worker(
        self,
        queue: asyncio.Queue[tuple[int, str, Callable[[], Awaitable[T]], int]],
        results: list[TaskResult[T] | None],
    ) -> None:
        while True:
            try:
                task_id, task_name, task_callable, attempt = await queue.get()
            except asyncio.CancelledError:
                break

            current_attempt = attempt + 1
            self._hooks.on_start(task_name, current_attempt)
            if self._cancel_event.is_set():
                results[task_id] = TaskResult(
                    name=task_name,
                    attempts=current_attempt,
                    cancelled=True,
                )
                self._hooks.on_cancel(task_name, current_attempt)
                queue.task_done()
                continue

            try:
                result = await task_callable()
            except asyncio.CancelledError:
                results[task_id] = TaskResult(
                    name=task_name,
                    attempts=current_attempt,
                    cancelled=True,
                )
                queue.task_done()
                self._hooks.on_cancel(task_name, current_attempt)
                raise
            except Exception as exc:
                if attempt < self._max_retries and not self._cancel_event.is_set():
                    self._hooks.on_retry(task_name, current_attempt, exc)
                    await queue.put((task_id, task_name, task_callable, attempt + 1))
                else:
                    results[task_id] = TaskResult(
                        name=task_name,
                        error=exc,
                        attempts=current_attempt,
                    )
                    self._hooks.on_failure(task_name, current_attempt, exc)
            else:
                results[task_id] = TaskResult(
                    name=task_name,
                    result=result,
                    attempts=current_attempt,
                )
                self._hooks.on_success(task_name, current_attempt)
            finally:
                queue.task_done()
