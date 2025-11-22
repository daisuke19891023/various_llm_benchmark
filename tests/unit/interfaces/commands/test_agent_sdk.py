"""CLI command tests for the OpenAI Agents SDK wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typer.testing import CliRunner

from various_llm_benchmark.interfaces import cli
from various_llm_benchmark.interfaces.commands import agent_sdk as agent_sdk_cmd
from various_llm_benchmark.models import ChatMessage, LLMResponse

if TYPE_CHECKING:
    import pytest

runner = CliRunner()


class StubAgentsProvider:
    """Stub that records invocations and returns predictable output."""

    def __init__(self) -> None:
        """Prepare call storage."""
        self.calls: list[dict[str, Any]] = []

    def complete(self, prompt: str) -> LLMResponse:
        """Simulate a completion call and record inputs."""
        self.calls.append({"kind": "complete", "prompt": prompt})
        return LLMResponse(content=f"agent-sdk:{prompt}", model="stub-model", raw={})

    def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        """Simulate a chat call and record inputs."""
        self.calls.append({"kind": "chat", "messages": messages})
        return LLMResponse(content=f"{len(messages)} messages", model="stub-model", raw={})


def test_agent_sdk_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Complete command should route to provider."""
    stub_provider = StubAgentsProvider()
    monkeypatch.setattr(agent_sdk_cmd, "_create_provider", lambda: stub_provider)

    result = runner.invoke(cli.app, ["agent-sdk", "complete", "hello"])

    assert result.exit_code == 0
    assert "agent-sdk:hello" in result.stdout
    assert stub_provider.calls == [{"kind": "complete", "prompt": "hello"}]


def test_agent_sdk_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chat command should build history and forward it."""
    stub_provider = StubAgentsProvider()
    monkeypatch.setattr(agent_sdk_cmd, "_create_provider", lambda: stub_provider)

    result = runner.invoke(
        cli.app,
        [
            "agent-sdk",
            "chat",
            "help me",
            "--history",
            "system:あなたは賢い",
        ],
    )

    assert result.exit_code == 0
    assert "3 messages" in result.stdout
    assert len(stub_provider.calls) == 1
    call = stub_provider.calls[0]
    assert call["kind"] == "chat"
    assert len(call["messages"]) == 3
    assert call["messages"][0].content.startswith("You are an assistant that helps")
    assert call["messages"][1].role == "system"
    assert call["messages"][2].content == "help me"
