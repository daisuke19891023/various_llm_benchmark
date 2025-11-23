"""Tests for Agno agent provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import various_llm_benchmark.agents.providers.agno as agno_module

from agno.models.anthropic import Claude
from agno.models.google import Gemini
from agno.models.message import Message
from agno.models.openai import OpenAIChat
from agno.run.agent import RunOutput

from various_llm_benchmark.agents.providers.agno import AgnoAgentProvider
from various_llm_benchmark.models import ChatMessage

if TYPE_CHECKING:
    from collections.abc import Callable
    import pytest


class DummyAgent:
    """Simple stub that records inputs passed to the agent."""

    def __init__(self, model: OpenAIChat | Claude | Gemini) -> None:
        """Store the model and prepare to capture inputs."""
        self.model = model
        self.inputs: list[Any] = []

    def run(self, run_input: Any, **_: Any) -> RunOutput:
        """Return a canned response and track the input."""
        self.inputs.append(run_input)
        output = RunOutput()
        output.model = self.model.id
        output.messages = [Message(role="assistant", content="dummy-response")]
        return output


def build_provider(factory: Callable[[OpenAIChat | Claude | Gemini], DummyAgent]) -> AgnoAgentProvider:
    """Construct provider with injected factory for determinism."""
    return AgnoAgentProvider(
        openai_api_key="openai-key",
        anthropic_api_key="anthropic-key",
        gemini_api_key="gemini-key",
        openai_model="gpt-test",
        anthropic_model="claude-test",
        gemini_model="gemini-test",
        temperature=0.5,
        agent_factory=factory,
    )


def test_complete_uses_openai_model_by_default(monkeypatch: Any) -> None:
    """Ensure OpenAI defaults and inputs are wired to the agent."""
    created: list[DummyAgent] = []
    times = iter([2.0, 3.0])
    monkeypatch.setattr(agno_module, "perf_counter", lambda: next(times))

    def factory(model: OpenAIChat | Claude | Gemini) -> DummyAgent:
        agent = DummyAgent(model)
        created.append(agent)
        return agent

    provider = build_provider(factory)

    response = provider.complete("Hello", provider="openai")

    assert response.content == "dummy-response"
    assert response.model == "gpt-test"
    assert response.elapsed_seconds == 1.0
    assert response.call_count == 1
    assert response.tool_calls == []
    assert len(created) == 1
    assert isinstance(created[0].model, OpenAIChat)
    assert created[0].inputs == ["Hello"]


def test_chat_appends_prompt_and_uses_history() -> None:
    """Convert chat history into agno messages and forward them."""
    created: list[DummyAgent] = []

    def factory(model: OpenAIChat | Claude | Gemini) -> DummyAgent:
        agent = DummyAgent(model)
        created.append(agent)
        return agent

    provider = build_provider(factory)

    messages = [
        ChatMessage(role="system", content="context"),
        ChatMessage(role="user", content="question1"),
        ChatMessage(role="assistant", content="answer1"),
        ChatMessage(role="user", content="next"),
    ]

    response = provider.chat(messages, provider="anthropic")

    assert response.content == "dummy-response"
    assert isinstance(created[0].model, Claude)
    recorded_messages = created[0].inputs[0]
    assert [(msg.role, msg.content) for msg in recorded_messages] == [
        ("system", "context"),
        ("user", "question1"),
        ("assistant", "answer1"),
        ("user", "next"),
    ]


def test_gemini_provider_builds_google_model() -> None:
    """Gemini provider selection should build Google Gemini model."""
    created: list[DummyAgent] = []

    def factory(model: OpenAIChat | Claude | Gemini) -> DummyAgent:
        agent = DummyAgent(model)
        created.append(agent)
        return agent

    provider = build_provider(factory)

    response = provider.complete("hello", provider="gemini", model="gemini-2.0")

    assert response.model == "gemini-2.0"
    assert isinstance(created[0].model, Gemini)
    assert created[0].model.temperature == 0.5


def test_default_agent_factory_applies_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default agent factory should pass instructions to the Agno Agent."""
    created_agents: list[StubAgent] = []

    class StubRunResult:
        def __init__(self, model_id: str) -> None:
            self.content = "ok"
            self.messages = None
            self.model = model_id

    class StubAgent:
        def __init__(
            self, model: OpenAIChat | Claude | Gemini, instructions: str | None = None,
        ) -> None:
            self.model = model
            self.instructions = instructions
            self.inputs: list[object] = []
            created_agents.append(self)

        def run(self, run_input: object, **_: object) -> StubRunResult:
            self.inputs.append(run_input)
            return StubRunResult(self.model.id)

    monkeypatch.setattr("various_llm_benchmark.agents.providers.agno.Agent", StubAgent)

    provider = AgnoAgentProvider(
        openai_api_key="openai-key",
        anthropic_api_key="anthropic-key",
        gemini_api_key="gemini-key",
        openai_model="gpt-test",
        anthropic_model="claude-test",
        gemini_model="gemini-test",
        temperature=0.5,
        instructions="follow steps",
    )

    response = provider.complete("Hello", provider="openai")

    assert response.content == "ok"
    assert created_agents[0].instructions == "follow steps"
    assert created_agents[0].inputs == ["Hello"]
