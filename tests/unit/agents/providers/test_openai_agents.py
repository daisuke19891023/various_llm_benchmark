"""Tests for OpenAI Agents SDK provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

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


def test_complete_builds_agent_and_runs_prompt() -> None:
    """Ensure prompts are forwarded and agent defaults are applied."""
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
