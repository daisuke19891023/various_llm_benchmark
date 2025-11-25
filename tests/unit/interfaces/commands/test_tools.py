from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from various_llm_benchmark.interfaces.cli import app
from various_llm_benchmark.llm.tools.retriever import RetrieverResponse
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
            return LLMResponse(
                content="Claude Opus 4.5 が最新モデルです。",
                model=model or "m",
                raw={"source": "test"},
            )

        return search

    monkeypatch.setattr(
        "various_llm_benchmark.interfaces.commands.tools.resolve_web_search_client",
        fake_resolver,
    )

    result = runner.invoke(
        app,
        [
            "tools",
            "web-search",
            "Claudeの最新モデルは?",
            "--model",
            "x",
            "--provider",
            "openai",
        ],
    )

    assert result.exit_code == 0
    assert captured == [("Claudeの最新モデルは?", "x")]
    assert "Claude Opus 4.5" in result.stdout


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
            return LLMResponse(
                content="Claude Opus 4.5 が最新モデルです。",
                model=model or "m",
                raw={"source": "test"},
            )

        return search

    monkeypatch.setattr(
        "various_llm_benchmark.interfaces.commands.tools.resolve_web_search_client",
        fake_resolver,
    )

    result = runner.invoke(
        app,
        [
            "tools",
            "web-search",
            "Claudeの最新モデルを教えて",
            "--provider",
            "anthropic",
            "--light-model",
        ],
    )

    assert result.exit_code == 0
    assert captured == [("Claudeの最新モデルを教えて", None)]
    assert "Claude Opus 4.5" in result.stdout


def test_tools_web_search_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini provider should route the request to the resolver."""
    captured: list[tuple[str, str | None]] = []

    def fake_resolver(
        provider: str,
        *,
        category: ToolCategory = ToolCategory.BUILTIN,
        use_light_model: bool = False,
    ) -> object:
        assert provider == "gemini"
        assert category is ToolCategory.BUILTIN
        assert use_light_model is True

        def search(prompt: str, model: str | None = None) -> LLMResponse:
            captured.append((prompt, model))
            return LLMResponse(
                content="Claude Opus 4.5 が最新モデルです。",
                model=model or "m",
                raw={"source": "test"},
            )

        return search

    monkeypatch.setattr(
        "various_llm_benchmark.interfaces.commands.tools.resolve_web_search_client",
        fake_resolver,
    )

    result = runner.invoke(
        app,
        [
            "tools",
            "web-search",
            "最新のClaudeを検索",
            "--provider",
            "gemini",
            "--light-model",
        ],
    )

    assert result.exit_code == 0
    assert captured == [("最新のClaudeを検索", None)]
    assert "Claude Opus 4.5" in result.stdout


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
        ) -> RetrieverResponse:
            captured.append(
                {
                    "query": query,
                    "model": model,
                    "top_k": top_k,
                    "threshold": threshold,
                    "timeout": timeout,
                },
            )
            return RetrieverResponse(documents=[])

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


def test_tools_shell_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """ShellコマンドがOpenAI指定で実行されることを確認する."""
    captured: list[tuple[str, list[str], float]] = []

    def fake_resolver(
        provider: str, *, category: ToolCategory = ToolCategory.BUILTIN,
    ) -> object:
        assert provider == "openai"
        assert category is ToolCategory.BUILTIN

        def execute(
            command: str,
            args: list[str] | None = None,
            *,
            timeout_seconds: float = 5.0,
        ) -> dict[str, object]:
            captured.append((command, args or [], timeout_seconds))
            return {"stdout": "README.md", "stderr": "", "exit_code": 0}

        return execute

    monkeypatch.setattr(
        "various_llm_benchmark.interfaces.commands.tools.resolve_shell_client",
        fake_resolver,
    )

    result = runner.invoke(
        app,
        [
            "tools",
            "shell",
            "ls",
            "--args",
            "-a",
            "--timeout-seconds",
            "1.5",
        ],
    )

    assert result.exit_code == 0
    assert captured == [("ls", ["-a"], 1.5)]
    assert '"exit_code": 0' in result.stdout


def test_tools_shell_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """ShellコマンドがClaude指定で実行されることを確認する."""
    captured: list[str] = []

    def fake_resolver(
        provider: str, *, category: ToolCategory = ToolCategory.BUILTIN,
    ) -> object:
        assert provider == "anthropic"
        assert category is ToolCategory.BUILTIN

        def execute(
            command: str,
            args: list[str] | None = None,
            *,
            timeout_seconds: float = 5.0,
        ) -> dict[str, object]:
            captured.append(command)
            return {
                "stdout": "src tests",
                "stderr": "",
                "exit_code": 0,
                "timeout": timeout_seconds,
                "args": args or [],
            }

        return execute

    monkeypatch.setattr(
        "various_llm_benchmark.interfaces.commands.tools.resolve_shell_client",
        fake_resolver,
    )

    result = runner.invoke(
        app,
        ["tools", "shell", "ls", "--provider", "anthropic"],
    )

    assert result.exit_code == 0
    assert captured == ["ls"]
    assert "src tests" in result.stdout
