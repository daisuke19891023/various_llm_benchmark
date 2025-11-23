from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from various_llm_benchmark.interfaces.cli import app
from various_llm_benchmark.interfaces.commands import dspy as dspy_commands
from various_llm_benchmark.models import ChatMessage, LLMResponse

if TYPE_CHECKING:
    import pytest

runner = CliRunner()


class _StubPromptTemplate:
    def __init__(self) -> None:
        self.system = "stub-system"

    def to_prompt_text(self, prompt: str) -> str:
        return f"{self.system}:{prompt}"


def test_dspy_complete_invokes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Complete should delegate to the DsPy client and echo the result."""
    captured: list[tuple[str, str | None]] = []

    def generate(prompt: str, model: str | None = None) -> LLMResponse:
        captured.append((prompt, model))
        return LLMResponse(content="done", model=model or "base", raw={"source": "test"})

    monkeypatch.setattr(dspy_commands, "_client", lambda: SimpleNamespace(generate=generate))
    monkeypatch.setattr(dspy_commands, "_prompt_template", lambda: _StubPromptTemplate())

    result = runner.invoke(app, ["dspy", "complete", "hello"], catch_exceptions=False)

    assert result.exit_code == 0
    assert result.stdout.strip() == "done"
    assert captured == [("stub-system:hello", None)]


def test_dspy_chat_invokes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chat should build messages and delegate to the DsPy client."""
    recorded_messages: list[list[ChatMessage]] = []

    def chat(messages: list[ChatMessage], model: str | None = None) -> LLMResponse:
        recorded_messages.append(messages)
        return LLMResponse(content=str(len(messages)), model=model or "base", raw={"source": "test"})

    monkeypatch.setattr(dspy_commands, "_client", lambda: SimpleNamespace(chat=chat))
    monkeypatch.setattr(dspy_commands, "_prompt_template", lambda: _StubPromptTemplate())

    result = runner.invoke(app, ["dspy", "chat", "hello"], catch_exceptions=False)

    assert result.exit_code == 0
    assert result.stdout.strip() == "2"
    assert recorded_messages[0][0] == ChatMessage(role="system", content="stub-system")
    assert recorded_messages[0][-1] == ChatMessage(role="user", content="hello")
