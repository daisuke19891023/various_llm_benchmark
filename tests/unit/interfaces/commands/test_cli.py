from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from various_llm_benchmark.interfaces.cli import app
from various_llm_benchmark.llm.protocol import LLMResponse

if TYPE_CHECKING:
    import pytest

runner = CliRunner()


def test_openai_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI complete command should print generated content."""

    def generate(prompt: str, model: str | None = None) -> LLMResponse:
        return LLMResponse(content=f"echo:{prompt}", model=model or "m", raw={"source": "test"})

    def fake_client() -> SimpleNamespace:
        return SimpleNamespace(generate=generate)

    monkeypatch.setattr("various_llm_benchmark.interfaces.commands.openai._client", fake_client)

    result = runner.invoke(app, ["openai", "complete", "hello"])

    assert result.exit_code == 0
    assert "echo:hello" in result.stdout


def test_openai_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI chat command should handle history and print response."""

    def chat(messages: list[object], model: str | None = None) -> LLMResponse:
        return LLMResponse(content=str(len(messages)), model=model or "m", raw={"source": "test"})

    def fake_client() -> SimpleNamespace:
        return SimpleNamespace(chat=chat)

    monkeypatch.setattr("various_llm_benchmark.interfaces.commands.openai._client", fake_client)

    result = runner.invoke(
        app,
        ["openai", "chat", "hi", "--history", "system:you are", "--history", "user:hey"],
    )

    assert result.exit_code == 0
    assert "3" in result.stdout  # 2 history + 1 prompt


def test_claude_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Claude complete command should print generated content."""

    def generate(prompt: str, model: str | None = None) -> LLMResponse:
        return LLMResponse(content=f"ok:{prompt}", model=model or "m", raw={"source": "test"})

    def fake_client() -> SimpleNamespace:
        return SimpleNamespace(generate=generate)

    monkeypatch.setattr("various_llm_benchmark.interfaces.commands.claude._client", fake_client)

    result = runner.invoke(app, ["claude", "complete", "ping"])

    assert result.exit_code == 0
    assert "ok:ping" in result.stdout
