from __future__ import annotations

import asyncio

from various_llm_benchmark.interfaces.runner.async_runner import AsyncJobRunner, TaskHooks


class RecordingHooks(TaskHooks):
    """Capture lifecycle events for verification."""

    def __init__(self) -> None:
        """Initialize an empty event list."""
        self.events: list[tuple[str, str, int, str | None]] = []

    def on_start(self, name: str, attempt: int) -> None:
        """Record a start event."""
        self.events.append(("start", name, attempt, None))

    def on_retry(self, name: str, attempt: int, error: BaseException) -> None:
        """Record a retry event with the error message."""
        self.events.append(("retry", name, attempt, str(error)))

    def on_success(self, name: str, attempt: int) -> None:
        """Record a successful completion event."""
        self.events.append(("success", name, attempt, None))

    def on_failure(self, name: str, attempt: int, error: BaseException) -> None:
        """Record a terminal failure event."""
        self.events.append(("failure", name, attempt, str(error)))

    def on_cancel(self, name: str, attempt: int) -> None:
        """Record a cancellation event."""
        self.events.append(("cancel", name, attempt, None))


def test_hooks_called_for_success_and_failure() -> None:
    """Runner invokes hooks for both successful and failing tasks."""
    hooks = RecordingHooks()
    runner: AsyncJobRunner[str] = AsyncJobRunner(concurrency=2, hooks=hooks)

    runner.add_task(lambda: "ok", name="success-task")

    def _fail() -> str:
        raise RuntimeError("expected failure")

    runner.add_task(_fail, name="failure-task")

    results = asyncio.run(runner.run())

    assert {result.name for result in results} == {"success-task", "failure-task"}
    assert ("success", "success-task", 1, None) in hooks.events
    assert any(event[0] == "failure" and event[1] == "failure-task" for event in hooks.events)


def test_retry_hook_invoked_before_success() -> None:
    """Runner reports retry before succeeding on a subsequent attempt."""
    hooks = RecordingHooks()
    runner: AsyncJobRunner[str] = AsyncJobRunner(concurrency=1, max_retries=1, hooks=hooks)

    attempts = {"count": 0}

    def _flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ValueError("boom")
        return "recovered"

    runner.add_task(_flaky, name="flaky-task")

    results = asyncio.run(runner.run())

    assert results[0].result == "recovered"
    assert ("retry", "flaky-task", 1, "boom") in hooks.events
    assert ("success", "flaky-task", 2, None) in hooks.events
