from __future__ import annotations

import asyncio

import pytest

from various_llm_benchmark.interfaces.runner import AsyncJobRunner


@pytest.mark.asyncio
async def test_runner_executes_all_tasks() -> None:
    """Ensure all tasks finish and preserve registration order."""
    runner = AsyncJobRunner[str](concurrency=2)
    completed: list[str] = []

    async def task(name: str, delay: float) -> str:
        await asyncio.sleep(delay)
        completed.append(name)
        return name

    runner.add_task(task, "a", 0.05, name="task-a")
    runner.add_task(task, "b", 0.01, name="task-b")
    runner.add_task(task, "c", 0.02, name="task-c")

    results = await runner.run()

    assert [result.result for result in results] == ["a", "b", "c"]
    assert completed == ["b", "c", "a"]
    assert all(result.succeeded for result in results)
    assert all(result.attempts == 1 for result in results)


@pytest.mark.asyncio
async def test_runner_retries_failures() -> None:
    """Ensure transient failures are retried up to the limit."""
    attempts = 0

    def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            msg = "temporary failure"
            raise RuntimeError(msg)
        return "ok"

    runner = AsyncJobRunner[str](concurrency=1, max_retries=2)
    runner.add_task(flaky, name="flaky")

    results = await runner.run()

    assert results[0].result == "ok"
    assert results[0].attempts == 2
    assert results[0].succeeded


@pytest.mark.asyncio
async def test_runner_cancels_remaining_tasks() -> None:
    """Ensure cancellation flag stops pending tasks."""
    runner = AsyncJobRunner[str](concurrency=1)

    async def slow_task(name: str) -> str:
        await asyncio.sleep(0.2)
        return name

    runner.add_task(slow_task, "first", name="slow-first")
    runner.add_task(slow_task, "second", name="slow-second")

    run_task = asyncio.create_task(runner.run())
    await asyncio.sleep(0.05)
    runner.request_cancel()

    results = await run_task

    assert any(result.cancelled for result in results)
    assert any(result.succeeded for result in results)


@pytest.mark.asyncio
async def test_runner_drains_queue_after_task_cancellation() -> None:
    """Cancelled tasks should not leave queued work hanging."""

    async def cancelled_task() -> str:
        await asyncio.sleep(0.01)
        raise asyncio.CancelledError

    runner = AsyncJobRunner[str](concurrency=1)
    runner.add_task(cancelled_task, name="cancelled")
    runner.add_task(lambda: "second", name="second")

    results = await asyncio.wait_for(runner.run(), timeout=1)

    assert len(results) == 2
    assert all(result.cancelled for result in results)
