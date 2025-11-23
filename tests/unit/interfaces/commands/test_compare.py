from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from various_llm_benchmark.interfaces.cli import app
from various_llm_benchmark.models import ChatMessage, LLMResponse

if TYPE_CHECKING:
    import pytest

runner = CliRunner()


def _client_stub(provider: str, recordings: list[dict[str, object]]) -> SimpleNamespace:
    def chat(
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        system_instruction: str | None = None,
    ) -> LLMResponse:
        recordings.append(
            {
                "provider": provider,
                "model": model,
                "system_instruction": system_instruction,
                "message_count": len(messages),
            },
        )
        return LLMResponse(content=f"{provider}:{model}", model=model or provider, raw={"provider": provider})

    return SimpleNamespace(chat=chat)


def test_compare_collects_all_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure all requested providers appear in the output."""
    recordings: list[dict[str, object]] = []
    factories = {
        "openai": lambda: _client_stub("openai", recordings),
        "anthropic": lambda: _client_stub("anthropic", recordings),
        "gemini": lambda: _client_stub("gemini", recordings),
    }
    monkeypatch.setattr(
        "various_llm_benchmark.interfaces.commands.compare.CLIENT_FACTORIES",
        factories,
    )

    result = runner.invoke(
        app,
        [
            "compare",
            "chat",
            "hello",
            "--target",
            "openai:o-model",
            "--target",
            "gemini:g-model",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert [entry["provider"] for entry in payload] == ["openai", "gemini"]
    assert payload[0]["content"] == "openai:o-model"
    assert payload[1]["content"] == "gemini:g-model"
    assert {record["model"] for record in recordings} == {"o-model", "g-model"}


def test_compare_handles_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure provider errors are surfaced without aborting the command."""

    def failing_factory() -> SimpleNamespace:
        def chat(
            _messages: list[ChatMessage],
            *,
            model: str | None = None,
            system_instruction: str | None = None,
        ) -> LLMResponse:
            _ = model
            _ = system_instruction
            raise RuntimeError("boom")

        return SimpleNamespace(chat=chat)

    factories = {
        "openai": lambda: _client_stub("openai", []),
        "anthropic": failing_factory,
    }
    monkeypatch.setattr(
        "various_llm_benchmark.interfaces.commands.compare.CLIENT_FACTORIES",
        factories,
    )

    result = runner.invoke(
        app,
        [
            "compare",
            "chat",
            "question",
            "--target",
            "openai:base",
            "--target",
            "claude:other",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert len(payload) == 2
    openai_entry = next(item for item in payload if item["provider"] == "openai")
    anthropic_entry = next(item for item in payload if item["provider"] == "claude")

    assert openai_entry["error"] is None
    assert anthropic_entry["error"] == "boom"
