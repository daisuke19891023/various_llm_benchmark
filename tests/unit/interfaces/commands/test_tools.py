from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from various_llm_benchmark.interfaces.cli import app
from various_llm_benchmark.models import LLMResponse

if TYPE_CHECKING:
    import pytest

runner = CliRunner()


def test_tools_web_search_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI provider should be used when selected."""
    captured: list[tuple[str, str | None]] = []

    def fake_openai_client() -> SimpleNamespace:
        def search(prompt: str, model: str | None = None) -> LLMResponse:
            captured.append((prompt, model))
            return LLMResponse(content="openai-result", model=model or "m", raw={"source": "test"})

        return SimpleNamespace(search=search)

    def fake_resolver(provider: str, *, use_light_model: bool = False) -> SimpleNamespace:
        assert provider == "openai"
        assert use_light_model is False
        return fake_openai_client()

    monkeypatch.setattr(
        "various_llm_benchmark.interfaces.commands.tools.resolve_web_search_client",
        fake_resolver,
    )

    result = runner.invoke(app, ["tools", "web-search", "find this", "--model", "x", "--provider", "openai"])

    assert result.exit_code == 0
    assert captured == [("find this", "x")]
    assert "openai-result" in result.stdout


def test_tools_web_search_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Claude provider should respond when selected."""
    captured: list[tuple[str, str | None]] = []

    def fake_anthropic_client() -> SimpleNamespace:
        def search(prompt: str, model: str | None = None) -> LLMResponse:
            captured.append((prompt, model))
            return LLMResponse(content="claude-result", model=model or "m", raw={"source": "test"})

        return SimpleNamespace(search=search)

    def fake_resolver(provider: str, *, use_light_model: bool = False) -> SimpleNamespace:
        assert provider == "anthropic"
        assert use_light_model is True
        return fake_anthropic_client()

    monkeypatch.setattr(
        "various_llm_benchmark.interfaces.commands.tools.resolve_web_search_client",
        fake_resolver,
    )

    result = runner.invoke(
        app, ["tools", "web-search", "docs", "--provider", "anthropic", "--light-model"],
    )

    assert result.exit_code == 0
    assert captured == [("docs", None)]
    assert "claude-result" in result.stdout
