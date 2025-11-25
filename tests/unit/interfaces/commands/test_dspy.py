from __future__ import annotations

from pathlib import Path
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


def test_dspy_optimize_runs_optimizer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Optimize should load dataset and report scores."""
    dataset = Path(tmp_path) / "dataset.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")

    def load_examples(path: Path) -> list[str]:
        assert path == dataset
        return ["example"]

    def optimize(examples: list[str], prompt_template: object, **kwargs: object) -> object:
        assert examples == ["example"]
        assert kwargs["max_bootstrapped_demos"] == 2
        assert kwargs["num_candidates"] == 3
        assert kwargs["num_threads"] == 1
        assert prompt_template is not None
        return SimpleNamespace(base_score=0.3, optimized_score=0.9, trainset_size=1)

    monkeypatch.setattr("various_llm_benchmark.llm.providers.dspy.optimizer.load_prompt_tuning_examples", load_examples)
    monkeypatch.setattr("various_llm_benchmark.llm.providers.dspy.optimizer.optimize_prompt", optimize)

    result = runner.invoke(
        app,
        ["dspy", "optimize", str(dataset), "--max-bootstrapped-demos", "2", "--num-candidates", "3"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "ベーススコア: 0.300" in result.stdout
    assert "最適化後スコア: 0.900" in result.stdout
