"""CLI command tests for the Agno agent wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typer.testing import CliRunner

from various_llm_benchmark.interfaces import cli
from various_llm_benchmark.interfaces.commands import agent as agent_cmd
from various_llm_benchmark.models import ChatMessage, LLMResponse

if TYPE_CHECKING:
    import pytest

runner = CliRunner()


class StubAgentProvider:
    """Stub that records invocations and returns predictable output."""

    def __init__(self) -> None:
        """Initialize stub storage."""
        self.calls: list[dict[str, Any]] = []

    def complete(
        self, prompt: str, *, provider: agent_cmd.ProviderName, model: str | None = None,
    ) -> LLMResponse:
        """Record a complete call and return canned response."""
        self.calls.append({"kind": "complete", "prompt": prompt, "provider": provider, "model": model})
        return LLMResponse(
            content=f"{provider}:{model or 'default'}:{prompt}",
            model=model or "stub-model",
            raw={},
        )

    def chat(
        self, messages: list[ChatMessage], *, provider: agent_cmd.ProviderName, model: str | None = None,
    ) -> LLMResponse:
        """Record a chat call and return count-based response."""
        self.calls.append(
            {
                "kind": "chat",
                "messages": messages,
                "provider": provider,
                "model": model,
            },
        )
        return LLMResponse(content=f"{len(messages)} messages", model=model or "stub-model", raw={})


def test_agent_complete_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Complete command should route to provider with given options."""
    stub_provider = StubAgentProvider()
    monkeypatch.setattr(agent_cmd, "_create_provider", lambda: stub_provider)

    result = runner.invoke(
        cli.app,
        ["agent", "complete", "hello", "--provider", "anthropic", "--model", "custom-model"],
    )

    assert result.exit_code == 0
    assert "anthropic:custom-model:hello" in result.stdout
    assert stub_provider.calls == [
        {"kind": "complete", "prompt": "hello", "provider": "anthropic", "model": "custom-model"},
    ]


def test_agent_chat_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chat command should assemble history and forward it."""
    stub_provider = StubAgentProvider()
    monkeypatch.setattr(agent_cmd, "_create_provider", lambda: stub_provider)

    result = runner.invoke(
        cli.app,
        [
            "agent",
            "chat",
            "help me",
            "--history",
            "system:あなたは賢い",
            "--provider",
            "openai",
        ],
    )

    assert result.exit_code == 0
    assert "3 messages" in result.stdout
    assert len(stub_provider.calls) == 1
    call = stub_provider.calls[0]
    assert call["kind"] == "chat"
    assert call["provider"] == "openai"
    assert len(call["messages"]) == 3
    assert call["messages"][0].content.startswith("You are an orchestration agent")
    assert call["messages"][1].role == "system"
    assert call["messages"][2].content == "help me"
