"""Tests for OpenAI Agents SDK provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, call

from various_llm_benchmark.agents.providers import openai_agents

from various_llm_benchmark.agents.providers.openai_agents import OpenAIAgentsProvider
from various_llm_benchmark.models import ChatMessage, LLMResponse

if TYPE_CHECKING:
    from agents import Agent
    from agents.result import RunResult


class StubRunResult:
    """Minimal RunResult replacement for tests."""

    def __init__(self, final_output: object) -> None:
        """Store the canned final output."""
        self.final_output = final_output
        self.new_items = []
        self.raw_responses = []


class RecordingRunFunction:
    """Records invocations for assertions."""

    def __init__(self, result: RunResult) -> None:
        """Initialize storage for calls and a prepared result."""
        self.calls: list[tuple[Agent, object]] = []
        self._result = result

    def __call__(self, agent: Agent, run_input: object) -> RunResult:
        """Capture invocation and return the prepared result."""
        self.calls.append((agent, run_input))
        return self._result


def test_complete_builds_agent_and_runs_prompt(monkeypatch: Any) -> None:
    """Ensure prompts are forwarded and agent defaults are applied."""
    times = iter([1.0, 1.4])
    monkeypatch.setattr(openai_agents, "perf_counter", lambda: next(times))
    run_function = RecordingRunFunction(cast("RunResult", StubRunResult("done")))
    provider = OpenAIAgentsProvider(
        api_key="dummy",
        model="gpt-4o-mini",
        instructions="be concise",
        temperature=0.2,
        run_function=run_function,
    )

    response = provider.complete("hello")

    assert isinstance(response, LLMResponse)
    assert response.content == "done"
    assert response.elapsed_seconds == 0.3999999999999999
    assert response.call_count == 1
    assert response.tool_calls == []
    assert len(run_function.calls) == 1
    agent, run_input = run_function.calls[0]
    assert run_input == "hello"
    assert agent.model == "gpt-4o-mini"
    assert agent.instructions == "be concise"
    assert agent.model_settings.temperature == 0.2


def test_chat_converts_messages() -> None:
    """History is converted into response input items and forwarded."""
    run_function = RecordingRunFunction(cast("RunResult", StubRunResult("final")))
    provider = OpenAIAgentsProvider(
        api_key="dummy",
        model="gpt-4o-mini",
        instructions="helpful",
        run_function=run_function,
    )
    messages = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="question"),
    ]

    response = provider.chat(messages)

    assert response.content == "final"
    assert len(run_function.calls) == 1
    _, run_input = run_function.calls[0]
    assert run_input == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "question"},
    ]


def test_complete_logs_structured_events(monkeypatch: Any) -> None:
    """OpenAI Agents provider should emit start, IO, and end logs."""
    times = iter([1.0, 2.0])
    monkeypatch.setattr(openai_agents, "perf_counter", lambda: next(times))

    logger = MagicMock()
    run_function = RecordingRunFunction(cast("RunResult", StubRunResult("done")))
    provider = OpenAIAgentsProvider(
        api_key="dummy",
        model="gpt-4o-mini",
        instructions="be concise",
        run_function=run_function,
    )
    provider.logger = logger

    provider.complete("ping")

    expected_calls = [
        call("start", action="openai_agents_complete", model="gpt-4o-mini"),
        call("io", direction="input", prompt="ping"),
        call("io", direction="output", model="gpt-4o-mini", content="done"),
        call("end", action="openai_agents_complete", elapsed_seconds=1.0),
    ]
    for expected in expected_calls:
        assert expected in logger.info.call_args_list
