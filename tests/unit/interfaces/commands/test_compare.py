from __future__ import annotations

import asyncio
import json

from various_llm_benchmark.interfaces.commands.compare import (
    CompareService,
    ComparisonResult,
    ComparisonTarget,
    results_to_json,
    summarize_results,
)
from various_llm_benchmark.interfaces.runner.async_runner import TaskHooks
from various_llm_benchmark.llm.protocol import LLMClient
from various_llm_benchmark.models import ChatMessage, LLMResponse, ToolCall


class RecordingHooks(TaskHooks):
    """Collect start and success events during comparisons."""

    def __init__(self) -> None:
        """Initialize event containers."""
        self.started: list[tuple[str, int]] = []
        self.succeeded: list[tuple[str, int]] = []

    def on_start(self, name: str, attempt: int) -> None:
        """Record a task start with attempt count."""
        self.started.append((name, attempt))

    def on_success(self, name: str, attempt: int) -> None:
        """Record a successful task completion."""
        self.succeeded.append((name, attempt))


class FakeClient(LLMClient):
    """Minimal client that echoes the prompt and model name."""

    def __init__(self, model_name: str) -> None:
        """Store the model name for later echoes."""
        self._model_name = model_name

    def chat(self, messages: list[ChatMessage], *, model: str | None = None) -> LLMResponse:
        """Return a simple LLMResponse using the last user prompt."""
        prompt = messages[-1].content
        return LLMResponse(
            content=f"{prompt} ({model or self._model_name})",
            model=model or self._model_name,
            raw={},
            elapsed_seconds=0.25,
            call_count=2,
            tool_calls=[ToolCall(name="search", arguments={"q": prompt}, output="done")],
        )

    def generate(self, prompt: str, *, model: str | None = None) -> LLMResponse:  # pragma: no cover - unused
        """Return a deterministic generation for testing."""
        return LLMResponse(content=prompt, model=model or self._model_name, raw={})

    def vision(
        self,
        prompt: str,
        image: object,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:  # pragma: no cover - unused
        """Return a deterministic vision response for testing."""
        _ = system_prompt
        return LLMResponse(
            content=f"{prompt} ({model or self._model_name}, image={image!r})",
            model=model or self._model_name,
            raw={},
        )


def test_compare_service_calls_hooks_and_returns_results() -> None:
    """CompareService should emit hook events and collate responses."""
    targets = [ComparisonTarget(provider="openai", model="test-model")]
    hooks = RecordingHooks()
    service = CompareService(
        prompt="hello",
        history=[ChatMessage(role="user", content="hi")],
        targets=targets,
        client_factories={"openai": lambda: FakeClient("default-model")},
    )

    results = asyncio.run(service.run_async(concurrency=1, hooks=hooks))

    assert results[0].provider == "openai"
    assert "test-model" in results[0].model
    assert "hello" in (results[0].content or "")
    assert results[0].call_count == 2
    assert results[0].elapsed_seconds == 0.25
    assert results[0].tools == ["search"]
    assert hooks.started == [("compare-openai", 1)]
    assert hooks.succeeded == [("compare-openai", 1)]


def test_summarize_results_accumulates_metrics() -> None:
    """Provider summaries should total elapsed time, calls, and tools."""
    results = [
        ComparisonResult(
            provider="openai",
            model="gpt",
            content="a",
            call_count=2,
            elapsed_seconds=1.0,
            tools=["search"],
        ),
        ComparisonResult(
            provider="openai",
            model="gpt",
            content="b",
            call_count=1,
            elapsed_seconds=0.5,
            tools=["retrieve"],
        ),
    ]

    summaries = summarize_results(results)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.total_elapsed_seconds == 1.5
    assert summary.total_call_count == 3
    assert summary.tools == ["retrieve", "search"]


def test_results_to_json_includes_summary() -> None:
    """JSON output includes a summary block."""
    results = [
        ComparisonResult(
            provider="gemini",
            model="gpt",
            content="ok",
            call_count=1,
            elapsed_seconds=0.4,
            tools=["search"],
        ),
    ]
    summaries = summarize_results(results)

    payload = json.loads(results_to_json(results, summaries))

    assert "results" in payload
    assert "summary" in payload
    assert payload["summary"][0]["total_call_count"] == 1
