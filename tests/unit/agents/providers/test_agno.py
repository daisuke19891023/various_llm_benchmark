"""Tests for Agno agent provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agno.models.anthropic import Claude
from agno.models.message import Message
from agno.models.openai import OpenAIChat
from agno.run.agent import RunOutput

from various_llm_benchmark.agents.providers.agno import AgnoAgentProvider
from various_llm_benchmark.llm.protocol import ChatMessage

if TYPE_CHECKING:
    from collections.abc import Callable


class DummyAgent:
    """Simple stub that records inputs passed to the agent."""

    def __init__(self, model: OpenAIChat | Claude) -> None:
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


def build_provider(factory: Callable[[OpenAIChat | Claude], DummyAgent]) -> AgnoAgentProvider:
    """Construct provider with injected factory for determinism."""
    return AgnoAgentProvider(
        openai_api_key="openai-key",
        anthropic_api_key="anthropic-key",
        openai_model="gpt-test",
        anthropic_model="claude-test",
        temperature=0.5,
        agent_factory=factory,
    )


def test_complete_uses_openai_model_by_default() -> None:
    """Ensure OpenAI defaults and inputs are wired to the agent."""
    created: list[DummyAgent] = []

    def factory(model: OpenAIChat | Claude) -> DummyAgent:
        agent = DummyAgent(model)
        created.append(agent)
        return agent

    provider = build_provider(factory)

    response = provider.complete("Hello", provider="openai")

    assert response.content == "dummy-response"
    assert response.model == "gpt-test"
    assert len(created) == 1
    assert isinstance(created[0].model, OpenAIChat)
    assert created[0].inputs == ["Hello"]


def test_chat_appends_prompt_and_uses_history() -> None:
    """Convert chat history into agno messages and forward them."""
    created: list[DummyAgent] = []

    def factory(model: OpenAIChat | Claude) -> DummyAgent:
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
