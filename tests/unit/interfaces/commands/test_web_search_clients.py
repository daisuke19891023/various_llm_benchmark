"""Web検索クライアントのディスパッチを検証するテスト."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from various_llm_benchmark.interfaces.commands import web_search_clients
from various_llm_benchmark.llm.tools.types import WebSearchInput
from various_llm_benchmark.models import LLMResponse

if TYPE_CHECKING:
    from collections.abc import Callable
    from various_llm_benchmark.interfaces.commands.web_search_clients import ProviderName


class _StubTool:
    """検索呼び出しの結果を固定で返すテスト用ツール."""

    def __init__(self, provider: str) -> None:
        self.provider = provider

    def search(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        return LLMResponse(
            content="Claude Opus 4.5 が最新です。",
            model=model or f"{self.provider}-model",
            raw={"provider": self.provider, "prompt": prompt},
        )


def _stub_builder(provider: str, captured: list[tuple[str, bool]]) -> Callable[..., _StubTool]:
    """キャプチャ付きのスタブビルダーを生成する."""

    def builder(*, use_light_model: bool = False) -> _StubTool:
        captured.append((provider, use_light_model))
        return _StubTool(provider)

    return builder


@pytest.mark.parametrize(
    ("provider", "use_light_model"),
    [
        ("openai", False),
        ("anthropic", True),
        ("gemini", False),
    ],
)
def test_resolve_web_search_client_dispatches_to_provider(
    provider: ProviderName,
    use_light_model: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """各プロバイダーで組み込み検索ツールを呼び出せることを確認する."""
    captured: list[tuple[str, bool]] = []
    web_search_clients.build_openai_web_search_tool.cache_clear()
    web_search_clients.build_anthropic_web_search_tool.cache_clear()
    web_search_clients.build_gemini_web_search_tool.cache_clear()

    monkeypatch.setattr(
        web_search_clients,
        "build_openai_web_search_tool",
        _stub_builder("openai", captured),
    )
    monkeypatch.setattr(
        web_search_clients,
        "build_anthropic_web_search_tool",
        _stub_builder("anthropic", captured),
    )
    monkeypatch.setattr(
        web_search_clients,
        "build_gemini_web_search_tool",
        _stub_builder("gemini", captured),
    )

    executor = web_search_clients.resolve_web_search_client(
        provider,
        use_light_model=use_light_model,
    )
    response = executor("Claudeの最新モデルは?", model="custom-model")

    assert response.content.startswith("Claude Opus 4.5")
    assert response.model == "custom-model"
    assert captured == [(provider, use_light_model)]


def test_web_search_executor_accepts_input_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """型付き入力モデルから検索を実行できる。."""
    captured: list[tuple[str, bool]] = []
    web_search_clients.build_openai_web_search_tool.cache_clear()
    web_search_clients.build_anthropic_web_search_tool.cache_clear()
    web_search_clients.build_gemini_web_search_tool.cache_clear()

    monkeypatch.setattr(
        web_search_clients,
        "build_openai_web_search_tool",
        _stub_builder("openai", captured),
    )
    monkeypatch.setattr(
        web_search_clients,
        "build_anthropic_web_search_tool",
        _stub_builder("anthropic", captured),
    )
    monkeypatch.setattr(
        web_search_clients,
        "build_gemini_web_search_tool",
        _stub_builder("gemini", captured),
    )

    executor = web_search_clients.resolve_web_search_client("openai", use_light_model=False)
    request = WebSearchInput(prompt="最新のGPTモデルは?", use_light_model=True)

    response = executor(request)

    assert response.model == "openai-model"
    assert response.raw == {"provider": "openai", "prompt": request.prompt}
    assert captured == [("openai", True)]


def test_web_search_input_schema_matches_model() -> None:
    """Web検索スキーマがPydanticモデルと同期している。."""
    assert WebSearchInput.model_json_schema() == web_search_clients.WEB_SEARCH_INPUT_SCHEMA
