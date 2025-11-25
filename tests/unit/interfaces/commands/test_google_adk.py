from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typer.testing import CliRunner

from various_llm_benchmark.interfaces import cli
from various_llm_benchmark.interfaces.commands import google_adk as google_adk_cmd
from various_llm_benchmark.models import ChatMessage, LLMResponse

if TYPE_CHECKING:
    import pytest

runner = CliRunner()


class StubADKProvider:
    """Stub that records ADK invocations."""

    def __init__(self) -> None:
        """Prepare storage for call records."""
        self.calls: list[dict[str, Any]] = []

    def complete(self, prompt: str) -> LLMResponse:
        """Simulate completion and record inputs."""
        self.calls.append({"kind": "complete", "prompt": prompt})
        return LLMResponse(content=f"adk:{prompt}", model="stub", raw={})

    def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        """Simulate chat and record provided messages."""
        self.calls.append({"kind": "chat", "messages": messages})
        return LLMResponse(content=f"{len(messages)} messages", model="stub", raw={})

    def vision(self, prompt: str, image: object) -> LLMResponse:
        """Simulate vision call and record payload."""
        self.calls.append({"kind": "vision", "prompt": prompt, "image": image})
        return LLMResponse(content="vision-ok", model="stub", raw={})


def test_google_adk_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Complete command should send prompt to provider."""
    stub_provider = StubADKProvider()

    def provider_factory(*, use_light_model: bool = False) -> StubADKProvider:
        assert use_light_model is False
        return stub_provider

    monkeypatch.setattr(google_adk_cmd, "_create_provider", provider_factory)

    result = runner.invoke(cli.app, ["google-adk", "complete", "hello"])

    assert result.exit_code == 0
    assert "adk:hello" in result.stdout
    assert stub_provider.calls == [{"kind": "complete", "prompt": "hello"}]


def test_google_adk_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chat command should build history messages."""
    stub_provider = StubADKProvider()

    def provider_factory(*, use_light_model: bool = False) -> StubADKProvider:
        assert use_light_model is False
        return stub_provider

    monkeypatch.setattr(google_adk_cmd, "_create_provider", provider_factory)

    result = runner.invoke(
        cli.app,
        ["google-adk", "chat", "help", "--history", "system:指示", "--history", "user:過去"],
    )

    assert result.exit_code == 0
    assert "4 messages" in result.stdout
    assert stub_provider.calls[0]["kind"] == "chat"
    assert stub_provider.calls[0]["messages"][0].role == "system"
    assert stub_provider.calls[0]["messages"][1].content == "指示"


def test_google_adk_web_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Web検索コマンドがGeminiツールを経由する."""
    captured: dict[str, Any] = {}

    def fake_builder(*, use_light_model: bool = False) -> Any:
        captured["use_light_model"] = use_light_model

        class Tool:
            def search(self, prompt: str, *, model: str | None = None) -> LLMResponse:
                captured["prompt"] = prompt
                captured["model"] = model
                return LLMResponse(content="adk-search", model=model or "stub", raw={})

        return Tool()

    monkeypatch.setattr(
        "various_llm_benchmark.interfaces.commands.web_search_clients.build_gemini_web_search_tool",
        fake_builder,
    )

    result = runner.invoke(cli.app, ["google-adk", "web-search", "topic", "--model", "gemini-3.0-pro"])

    assert result.exit_code == 0
    assert captured == {"use_light_model": False, "prompt": "topic", "model": "gemini-3.0-pro"}
    assert "adk-search" in result.stdout
