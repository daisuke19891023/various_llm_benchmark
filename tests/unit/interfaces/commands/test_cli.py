from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from various_llm_benchmark.interfaces.cli import app
from various_llm_benchmark.models import ChatMessage, LLMResponse

if TYPE_CHECKING:
    import pytest

runner = CliRunner()


def test_openai_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI complete command should print generated content."""
    captured: list[str] = []

    def generate(prompt: str, model: str | None = None) -> LLMResponse:
        captured.append(prompt)
        return LLMResponse(content=f"echo:{prompt}", model=model or "m", raw={"source": "test"})

    def fake_client() -> SimpleNamespace:
        return SimpleNamespace(generate=generate)

    monkeypatch.setattr("various_llm_benchmark.interfaces.commands.openai._client", fake_client)

    result = runner.invoke(app, ["openai", "complete", "hello"])

    assert result.exit_code == 0
    assert "echo:" in result.stdout
    assert captured[0].startswith("You are a helpful assistant.")
    assert captured[0].rstrip().endswith("hello")


def test_openai_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI chat command should handle history and print response."""

    def chat(messages: list[ChatMessage], model: str | None = None) -> LLMResponse:
        return LLMResponse(content=str(len(messages)), model=model or "m", raw={"source": "test"})

    def fake_client() -> SimpleNamespace:
        return SimpleNamespace(chat=chat)

    monkeypatch.setattr("various_llm_benchmark.interfaces.commands.openai._client", fake_client)

    result = runner.invoke(
        app,
        ["openai", "chat", "hi", "--history", "system:you are", "--history", "user:hey"],
    )

    assert result.exit_code == 0
    assert "4" in result.stdout  # system prompt + 2 history + 1 user prompt


def test_claude_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Claude complete command should print generated content."""
    captured: list[str] = []

    def generate(prompt: str, model: str | None = None) -> LLMResponse:
        captured.append(prompt)
        return LLMResponse(content=f"ok:{prompt}", model=model or "m", raw={"source": "test"})

    def fake_client() -> SimpleNamespace:
        return SimpleNamespace(generate=generate)

    monkeypatch.setattr("various_llm_benchmark.interfaces.commands.claude._client", fake_client)

    result = runner.invoke(app, ["claude", "complete", "ping"])

    assert result.exit_code == 0
    assert "ok:" in result.stdout
    assert captured[0].startswith("You are a clear and concise assistant")
    assert captured[0].rstrip().endswith("ping")


def test_claude_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Claude chat command should include system prompts."""
    recorded_messages: list[list[ChatMessage]] = []

    def chat(messages: list[ChatMessage], model: str | None = None) -> LLMResponse:
        recorded_messages.append(messages)
        return LLMResponse(content=str(len(messages)), model=model or "m", raw={"source": "test"})

    def fake_client() -> SimpleNamespace:
        return SimpleNamespace(chat=chat)

    monkeypatch.setattr("various_llm_benchmark.interfaces.commands.claude._client", fake_client)

    result = runner.invoke(app, ["claude", "chat", "hi"], catch_exceptions=False)

    assert result.exit_code == 0
    assert recorded_messages[0][0].role == "system"
    assert recorded_messages[0][-1].content == "hi"
    assert result.stdout.strip() == "2"
