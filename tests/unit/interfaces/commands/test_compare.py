from __future__ import annotations

import asyncio

from various_llm_benchmark.interfaces.commands.compare import (
    CompareService,
    ComparisonTarget,
)
from various_llm_benchmark.interfaces.runner.async_runner import TaskHooks
from various_llm_benchmark.models import ChatMessage, LLMResponse
from various_llm_benchmark.llm.protocol import LLMClient


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
        return LLMResponse(content=f"{prompt} ({model or self._model_name})", model=model or self._model_name, raw={})

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
    assert hooks.started == [("compare-openai", 1)]
    assert hooks.succeeded == [("compare-openai", 1)]
