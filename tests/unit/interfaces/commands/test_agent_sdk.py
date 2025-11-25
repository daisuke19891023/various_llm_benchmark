"""CLI command tests for the OpenAI Agents SDK wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typer.testing import CliRunner

from various_llm_benchmark.interfaces import cli
from various_llm_benchmark.interfaces.commands import agent_sdk as agent_sdk_cmd
from various_llm_benchmark.llm.tools.retriever import RetrieverResponse
from various_llm_benchmark.llm.tools.registry import ToolCategory
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

    def provider_factory(*, use_light_model: bool = False) -> StubAgentsProvider:
        assert use_light_model is False
        return stub_provider

    monkeypatch.setattr(agent_sdk_cmd, "_create_provider", provider_factory)

    result = runner.invoke(cli.app, ["agent-sdk", "complete", "hello"])

    assert result.exit_code == 0
    assert "agent-sdk:hello" in result.stdout
    assert stub_provider.calls == [{"kind": "complete", "prompt": "hello"}]


def test_agent_sdk_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chat command should build history and forward it."""
    stub_provider = StubAgentsProvider()

    def provider_factory(*, use_light_model: bool = False) -> StubAgentsProvider:
        assert use_light_model is False
        return stub_provider

    monkeypatch.setattr(agent_sdk_cmd, "_create_provider", provider_factory)

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


def test_agent_sdk_web_search_uses_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Web検索コマンドがresolver経由で検索を実行する."""
    captured: dict[str, Any] = {}

    def fake_resolver(
        provider: str,
        *,
        category: ToolCategory = ToolCategory.BUILTIN,
        use_light_model: bool = False,
    ) -> object:
        captured["provider"] = provider
        captured["category"] = category
        captured["use_light_model"] = use_light_model

        def search(prompt: str, model: str | None = None) -> LLMResponse:
            captured["prompt"] = prompt
            captured["model"] = model
            return LLMResponse(content="web-sdk", model=model or "stub", raw={})

        return search

    monkeypatch.setattr(agent_sdk_cmd, "resolve_web_search_client", fake_resolver)

    result = runner.invoke(
        cli.app,
        ["agent-sdk", "web-search", "topic", "--model", "gpt-5.1", "--light-model"],
    )

    assert result.exit_code == 0
    assert captured == {
        "provider": "openai",
        "category": ToolCategory.BUILTIN,
        "use_light_model": True,
        "prompt": "topic",
        "model": "gpt-5.1",
    }
    assert "web-sdk" in result.stdout


def test_agent_sdk_retriever_uses_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrieverコマンドがresolver経由で呼び出されることを確認する."""
    captured: list[dict[str, object]] = []

    def fake_resolver(
        provider: str,
        *,
        category: ToolCategory = ToolCategory.BUILTIN,
    ) -> object:
        assert provider == "openai"
        assert category is ToolCategory.BUILTIN

        def retrieve(
            query: str,
            *,
            model: str | None = None,
            top_k: int | None = None,
            threshold: float | None = None,
            timeout: float = 5.0,
        ) -> RetrieverResponse:
            captured.append(
                {
                    "query": query,
                    "model": model,
                    "top_k": top_k,
                    "threshold": threshold,
                    "timeout": timeout,
                },
            )
            return RetrieverResponse(documents=[])

        return retrieve

    monkeypatch.setattr(agent_sdk_cmd, "resolve_retriever_client", fake_resolver)

    result = runner.invoke(
        cli.app,
        [
            "agent-sdk",
            "retriever",
            "topic",
            "--model",
            "embed-v1",
            "--top-k",
            "6",
            "--threshold",
            "0.4",
            "--timeout",
            "2.0",
        ],
    )

    assert result.exit_code == 0
    assert captured == [
        {
            "query": "topic",
            "model": "embed-v1",
            "top_k": 6,
            "threshold": 0.4,
            "timeout": 2.0,
        },
    ]
    assert "documents" in result.stdout
