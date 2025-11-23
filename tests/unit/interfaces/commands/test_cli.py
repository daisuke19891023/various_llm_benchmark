from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from various_llm_benchmark.interfaces.cli import app
from various_llm_benchmark.models import ChatMessage, LLMResponse
from various_llm_benchmark.settings import settings

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


def test_gemini_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini complete command should apply provider prompt."""
    captured: list[str] = []

    def generate(prompt: str, model: str | None = None) -> LLMResponse:
        captured.append(prompt)
        return LLMResponse(content=f"gm:{prompt}", model=model or "g", raw={"source": "test"})

    def fake_client() -> SimpleNamespace:
        return SimpleNamespace(generate=generate)

    monkeypatch.setattr("various_llm_benchmark.interfaces.commands.gemini._client", fake_client)

    result = runner.invoke(app, ["gemini", "complete", "hey"])

    assert result.exit_code == 0
    assert "gm:" in result.stdout
    assert captured[0].startswith("You are a concise and reliable assistant powered by Gemini")
    assert captured[0].rstrip().endswith("hey")


def test_dspy_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """DsPy complete command should include provider prompt."""
    captured: list[str] = []

    def generate(prompt: str, model: str | None = None) -> LLMResponse:
        captured.append(prompt)
        return LLMResponse(content=f"ds:{prompt}", model=model or "d", raw={"source": "test"})

    def fake_client() -> SimpleNamespace:
        return SimpleNamespace(generate=generate)

    monkeypatch.setattr("various_llm_benchmark.interfaces.commands.dspy._client", fake_client)

    result = runner.invoke(app, ["dspy", "complete", "hello"])

    assert result.exit_code == 0
    assert "ds:" in result.stdout
    assert captured[0].startswith("You are a succinct assistant powered by DsPy")
    assert captured[0].rstrip().endswith("hello")


def test_dspy_client_passes_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """DsPy client factory should forward the configured API key."""
    captured: dict[str, object] = {}

    class RecordingClient(SimpleNamespace):
        def __init__(
            self, default_model: str, *, temperature: float, **kwargs: object,
        ) -> None:
            super().__init__()
            captured["model"] = default_model
            captured["temperature"] = temperature
            captured["kwargs"] = kwargs

            self.model = default_model

        def generate(self, prompt: str, model: str | None = None) -> LLMResponse:
            return LLMResponse(
                content=f"ok:{prompt}", model=model or self.model, raw={"source": "test"},
            )

    monkeypatch.setattr(
        "various_llm_benchmark.interfaces.commands.dspy.DsPyLLMClient",
        RecordingClient,
    )

    from various_llm_benchmark.interfaces.commands.dspy import dspy_complete

    dspy_complete("hello")

    assert captured["model"] == settings.openai_model
    assert captured["temperature"] == settings.default_temperature
    expected_key = settings.openai_api_key.get_secret_value()
    assert captured["kwargs"] == {"api_key": expected_key}


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


def test_gemini_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini chat command should include provider system prompt."""
    recorded_messages: list[list[ChatMessage]] = []
    recorded_system: list[str | None] = []

    def chat(
        messages: list[ChatMessage], *, model: str | None = None, system_instruction: str | None = None,
    ) -> LLMResponse:
        recorded_messages.append(messages)
        recorded_system.append(system_instruction)
        return LLMResponse(content=str(len(messages)), model=model or "g", raw={"source": "test"})

    def fake_client() -> SimpleNamespace:
        return SimpleNamespace(chat=chat)

    monkeypatch.setattr("various_llm_benchmark.interfaces.commands.gemini._client", fake_client)

    result = runner.invoke(app, ["gemini", "chat", "hi"], catch_exceptions=False)

    assert result.exit_code == 0
    assert recorded_messages[0][0].role == "user"
    assert recorded_messages[0][-1].content == "hi"
    assert recorded_system[0] is not None
    assert recorded_system[0].startswith("You are a concise and reliable assistant powered by Gemini")
    assert result.stdout.strip() == "1"


def test_dspy_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    """DsPy chat command should include system prompt when building messages."""
    recorded_messages: list[list[ChatMessage]] = []

    def chat(messages: list[ChatMessage], model: str | None = None) -> LLMResponse:
        recorded_messages.append(messages)
        return LLMResponse(content=str(len(messages)), model=model or "d", raw={"source": "test"})

    def fake_client() -> SimpleNamespace:
        return SimpleNamespace(chat=chat)

    monkeypatch.setattr("various_llm_benchmark.interfaces.commands.dspy._client", fake_client)

    result = runner.invoke(app, ["dspy", "chat", "hi"], catch_exceptions=False)

    assert result.exit_code == 0
    assert recorded_messages[0][0].role == "system"
    assert recorded_messages[0][-1].content == "hi"
    assert result.stdout.strip() == "2"
