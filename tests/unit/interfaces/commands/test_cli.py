from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from various_llm_benchmark.interfaces.cli import app
from various_llm_benchmark.models import ChatMessage, LLMResponse, MediaInput

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()


def test_openai_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI complete command should print generated content."""
    captured: list[str] = []
    captured_reasoning: list[str | None] = []
    captured_verbosity: list[str | None] = []

    def generate(
        prompt: str,
        model: str | None = None,
        reasoning_effort: str | None = None,
        verbosity: str | None = None,
    ) -> LLMResponse:
        captured.append(prompt)
        captured_reasoning.append(reasoning_effort)
        captured_verbosity.append(verbosity)
        return LLMResponse(content=f"echo:{prompt}", model=model or "m", raw={"source": "test"})

    def fake_client() -> SimpleNamespace:
        return SimpleNamespace(generate=generate)

    monkeypatch.setattr("various_llm_benchmark.interfaces.commands.openai._client", fake_client)

    result = runner.invoke(app, ["openai", "complete", "hello"])

    assert result.exit_code == 0
    assert "echo:" in result.stdout
    assert captured[0].startswith("You are a helpful assistant.")
    assert captured[0].rstrip().endswith("hello")
    assert captured_reasoning[0] is None
    assert captured_verbosity[0] is None


def test_openai_complete_with_reasoning_and_verbosity(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI complete command should forward reasoning options."""
    recorded: dict[str, str | None] = {}

    def generate(
        prompt: str,
        model: str | None = None,
        reasoning_effort: str | None = None,
        verbosity: str | None = None,
    ) -> LLMResponse:
        recorded["prompt"] = prompt
        recorded["reasoning_effort"] = reasoning_effort
        recorded["verbosity"] = verbosity
        return LLMResponse(content="ok", model=model or "m", raw={"source": "test"})

    def fake_client() -> SimpleNamespace:
        return SimpleNamespace(generate=generate)

    monkeypatch.setattr("various_llm_benchmark.interfaces.commands.openai._client", fake_client)

    result = runner.invoke(
        app,
        [
            "openai",
            "complete",
            "hello",
            "--reasoning-effort",
            "high",
            "--verbosity",
            "high",
        ],
    )

    assert result.exit_code == 0
    assert recorded["reasoning_effort"] == "high"
    assert recorded["verbosity"] == "high"
    assert recorded["prompt"] is not None


def test_gemini_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini complete command should apply provider prompt."""
    captured: list[str] = []
    recorded_thinking: list[str | None] = []

    def generate(prompt: str, model: str | None = None, thinking_level: str | None = None) -> LLMResponse:
        captured.append(prompt)
        recorded_thinking.append(thinking_level)
        return LLMResponse(content=f"gm:{prompt}", model=model or "g", raw={"source": "test"})

    def fake_client() -> SimpleNamespace:
        return SimpleNamespace(generate=generate)

    monkeypatch.setattr("various_llm_benchmark.interfaces.commands.gemini._client", fake_client)

    result = runner.invoke(app, ["gemini", "complete", "hey"])

    assert result.exit_code == 0
    assert "gm:" in result.stdout
    assert captured[0].startswith("You are a concise and reliable assistant powered by Gemini")
    assert captured[0].rstrip().endswith("hey")
    assert recorded_thinking[0] is None


def test_gemini_complete_accepts_thinking_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini complete should forward thinking level option."""
    recorded: dict[str, object] = {}

    def generate(prompt: str, model: str | None = None, thinking_level: str | None = None) -> LLMResponse:
        recorded.update({"prompt": prompt, "model": model, "thinking_level": thinking_level})
        return LLMResponse(content=f"gm:{prompt}", model=model or "g", raw={"source": "test"})

    def fake_client() -> SimpleNamespace:
        return SimpleNamespace(generate=generate)

    monkeypatch.setattr("various_llm_benchmark.interfaces.commands.gemini._client", fake_client)

    result = runner.invoke(app, ["gemini", "complete", "hey", "--thinking-level", "high"])

    assert result.exit_code == 0
    assert recorded["thinking_level"] == "high"
    assert recorded["prompt"] is not None


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


def test_openai_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI chat command should handle history and print response."""
    recorded_reasoning: list[str | None] = []
    recorded_verbosity: list[str | None] = []

    def chat(
        messages: list[ChatMessage],
        model: str | None = None,
        reasoning_effort: str | None = None,
        verbosity: str | None = None,
    ) -> LLMResponse:
        recorded_reasoning.append(reasoning_effort)
        recorded_verbosity.append(verbosity)
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
    assert recorded_reasoning[0] is None
    assert recorded_verbosity[0] is None


def test_openai_chat_with_reasoning_and_verbosity(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI chat command should pass reasoning settings to the client."""
    recorded: dict[str, str | None] = {}

    def chat(
        messages: list[ChatMessage],
        model: str | None = None,
        reasoning_effort: str | None = None,
        verbosity: str | None = None,
    ) -> LLMResponse:
        recorded["reasoning_effort"] = reasoning_effort
        recorded["verbosity"] = verbosity
        recorded["count"] = str(len(messages))
        return LLMResponse(content="ok", model=model or "m", raw={"source": "test"})

    def fake_client() -> SimpleNamespace:
        return SimpleNamespace(chat=chat)

    monkeypatch.setattr("various_llm_benchmark.interfaces.commands.openai._client", fake_client)

    result = runner.invoke(
        app,
        [
            "openai",
            "chat",
            "hi",
            "--reasoning-effort",
            "medium",
            "--verbosity",
            "low",
        ],
    )

    assert result.exit_code == 0
    assert recorded["reasoning_effort"] == "medium"
    assert recorded["verbosity"] == "low"
    assert recorded["count"] == "2"


def test_claude_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Claude complete command should print generated content."""
    captured: list[str] = []

    def generate(
        prompt: str,
        model: str | None = None,
        *,
        thinking: dict[str, object] | None = None,
    ) -> LLMResponse:
        captured.append(prompt)
        if thinking is not None:
            captured.append(str(thinking))
        return LLMResponse(content=f"ok:{prompt}", model=model or "m", raw={"source": "test"})

    def fake_client(extended_thinking: bool = False) -> SimpleNamespace:
        return SimpleNamespace(generate=generate)

    monkeypatch.setattr("various_llm_benchmark.interfaces.commands.claude._client", fake_client)

    result = runner.invoke(
        app,
        ["claude", "complete", "ping", "--extended-thinking", "--thinking-tokens", "9000"],
    )

    assert result.exit_code == 0
    assert "ok:" in result.stdout
    assert captured[0].startswith("You are a clear and concise assistant")
    assert captured[0].rstrip().endswith("ping")
    assert "'budget_tokens': 9000" in captured[1]


def test_claude_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Claude chat command should include system prompts."""
    recorded_messages: list[list[ChatMessage]] = []
    recorded_thinking: list[dict[str, object] | None] = []

    def chat(
        messages: list[ChatMessage],
        model: str | None = None,
        *,
        thinking: dict[str, object] | None = None,
    ) -> LLMResponse:
        recorded_messages.append(messages)
        recorded_thinking.append(thinking)
        return LLMResponse(content=str(len(messages)), model=model or "m", raw={"source": "test"})

    def fake_client(extended_thinking: bool = False) -> SimpleNamespace:
        return SimpleNamespace(chat=chat)

    monkeypatch.setattr("various_llm_benchmark.interfaces.commands.claude._client", fake_client)

    result = runner.invoke(
        app,
        ["claude", "chat", "hi", "--extended-thinking", "--thinking-tokens", "7000"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    # Claude chat doesn't include system prompt in messages (it's passed separately to API)
    assert recorded_messages[0][0].role == "user"
    assert recorded_messages[0][-1].content == "hi"
    assert recorded_thinking[0] == {"type": "enabled", "budget_tokens": 7000}
    assert result.stdout.strip() == "1"


def test_gemini_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini chat command should include provider system prompt."""
    recorded_messages: list[list[ChatMessage]] = []
    recorded_system: list[str | None] = []
    recorded_thinking: list[str | None] = []

    def chat(
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        system_instruction: str | None = None,
        thinking_level: str | None = None,
    ) -> LLMResponse:
        recorded_messages.append(messages)
        recorded_system.append(system_instruction)
        recorded_thinking.append(thinking_level)
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
    assert recorded_thinking[0] is None


def test_gemini_chat_accepts_thinking_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini chat should pass thinking level when requested."""
    recorded: dict[str, object] = {}

    def chat(
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        system_instruction: str | None = None,
        thinking_level: str | None = None,
    ) -> LLMResponse:
        recorded.update(
            {
                "messages": messages,
                "system_instruction": system_instruction,
                "thinking_level": thinking_level,
            },
        )
        return LLMResponse(content=str(len(messages)), model=model or "g", raw={"source": "test"})

    def fake_client() -> SimpleNamespace:
        return SimpleNamespace(chat=chat)

    monkeypatch.setattr("various_llm_benchmark.interfaces.commands.gemini._client", fake_client)

    result = runner.invoke(
        app,
        [
            "gemini",
            "chat",
            "hi",
            "--history",
            "user:prev",
            "--thinking-level",
            "high",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert recorded["thinking_level"] == "high"
    assert isinstance(recorded.get("messages"), list)


def test_gemini_multimodal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Gemini multimodal command should load media and apply system prompt."""
    recorded_media: list[list[MediaInput]] = []
    recorded_system: list[str | None] = []
    recorded_thinking: list[str | None] = []

    def multimodal(
        prompt: str,
        media: list[MediaInput],
        *,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> LLMResponse:
        del prompt
        recorded_media.append(media)
        recorded_thinking.append(thinking_level)
        return LLMResponse(content=str(len(media)), model=model or "g", raw={"source": "test"})

    def fake_client() -> SimpleNamespace:
        return SimpleNamespace(multimodal=multimodal)

    def fake_reader(path: Path) -> MediaInput:
        return MediaInput(media_type="audio/wav", data=path.name)

    monkeypatch.setattr("various_llm_benchmark.interfaces.commands.gemini._client", fake_client)
    monkeypatch.setattr(
        "various_llm_benchmark.interfaces.commands.gemini.read_audio_or_video_file",
        fake_reader,
    )

    media_file = tmp_path / "sound.wav"
    media_file.write_bytes(b"data")

    result = runner.invoke(
        app,
        ["gemini", "multimodal", "hi", str(media_file), "--thinking-level", "base"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert recorded_media[0][0].data == "sound.wav"
    assert recorded_thinking[0] == "base"
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
