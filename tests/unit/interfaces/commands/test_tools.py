from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from various_llm_benchmark.interfaces.cli import app
from various_llm_benchmark.llm.tools.registry import ToolCategory
from various_llm_benchmark.models import LLMResponse

if TYPE_CHECKING:
    import pytest

runner = CliRunner()


def test_tools_web_search_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI provider should be used when selected."""
    captured: list[tuple[str, str | None]] = []

    def fake_resolver(
        provider: str,
        *,
        category: ToolCategory = ToolCategory.BUILTIN,
        use_light_model: bool = False,
    ) -> object:
        assert provider == "openai"
        assert category is ToolCategory.BUILTIN
        assert use_light_model is False

        def search(prompt: str, model: str | None = None) -> LLMResponse:
            captured.append((prompt, model))
            return LLMResponse(content="openai-result", model=model or "m", raw={"source": "test"})

        return search

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

    def fake_resolver(
        provider: str,
        *,
        category: ToolCategory = ToolCategory.BUILTIN,
        use_light_model: bool = False,
    ) -> object:
        assert provider == "anthropic"
        assert category is ToolCategory.BUILTIN
        assert use_light_model is True

        def search(prompt: str, model: str | None = None) -> LLMResponse:
            captured.append((prompt, model))
            return LLMResponse(content="claude-result", model=model or "m", raw={"source": "test"})

        return search

    monkeypatch.setattr(
        "various_llm_benchmark.interfaces.commands.tools.resolve_web_search_client",
        fake_resolver,
    )

    result = runner.invoke(
        app,
        ["tools", "web-search", "docs", "--provider", "anthropic", "--light-model"],
    )

    assert result.exit_code == 0
    assert captured == [("docs", None)]
    assert "claude-result" in result.stdout


def test_tools_retriever(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrieverサブコマンドがresolver経由で呼び出されることを確認する."""
    captured: list[dict[str, object]] = []

    def fake_resolver(
        provider: str,
        *,
        category: ToolCategory = ToolCategory.BUILTIN,
    ) -> object:
        assert provider == "google"
        assert category is ToolCategory.BUILTIN

        def retrieve(
            query: str,
            *,
            model: str | None = None,
            top_k: int | None = None,
            threshold: float | None = None,
            timeout: float = 5.0,
        ) -> dict[str, object]:
            captured.append(
                {
                    "query": query,
                    "model": model,
                    "top_k": top_k,
                    "threshold": threshold,
                    "timeout": timeout,
                },
            )
            return {"documents": []}

        return retrieve

    monkeypatch.setattr(
        "various_llm_benchmark.interfaces.commands.tools.resolve_retriever_client",
        fake_resolver,
    )

    result = runner.invoke(
        app,
        [
            "tools",
            "retriever",
            "query text",
            "--provider",
            "google",
            "--model",
            "embed-v1",
            "--top-k",
            "7",
            "--threshold",
            "0.3",
            "--timeout",
            "1.5",
        ],
    )

    assert result.exit_code == 0
    assert captured == [
        {
            "query": "query text",
            "model": "embed-v1",
            "top_k": 7,
            "threshold": 0.3,
            "timeout": 1.5,
        },
    ]
    assert "documents" in result.stdout
