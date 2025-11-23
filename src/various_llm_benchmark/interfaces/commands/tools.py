from __future__ import annotations

import json
from typing import Literal

import typer

from various_llm_benchmark.interfaces.commands.retriever_clients import (
    resolve_retriever_client,
)
from various_llm_benchmark.interfaces.commands.web_search_clients import resolve_web_search_client
from various_llm_benchmark.llm.tools.registry import ToolCategory

ProviderName = Literal["openai", "anthropic", "gemini"]


tools_app = typer.Typer(help="LLMの組み込みツール呼び出しを実行します。")

PROVIDER_OPTION: ProviderName = typer.Option(
    "openai",
    "--provider",
    "-p",
    case_sensitive=False,
    help="利用するプロバイダー (openai / anthropic / gemini)",
)

MODEL_OPTION: str | None = typer.Option(default=None, help="モデル名を上書きします。")
LIGHT_MODEL_OPTION: bool = typer.Option(
    default=False,
    help="軽量モデル (gpt-5.1-mini / claude-4.5-haiku / gemini-2.5-flash) を使用します。",
)
CATEGORY_OPTION: ToolCategory = typer.Option(
    default=ToolCategory.BUILTIN,
    case_sensitive=False,
    help="利用するツールカテゴリ (builtin / mcp / external)",
)
THRESHOLD_OPTION: float | None = typer.Option(
    None,
    help="スコアのしきい値 (0-1)。未指定の場合は設定値を使用します。",
    min=0.0,
    max=1.0,
)
TOP_K_OPTION: int | None = typer.Option(
    None,
    help="返却件数の上限。未指定の場合は設定値を使用します。",
    min=1,
)
TIMEOUT_OPTION: float = typer.Option(
    5.0,
    help="DB検索や埋め込み取得のタイムアウト (秒)",
    min=0.0,
)


@tools_app.command("web-search")
def web_search(
    prompt: str,
    provider: ProviderName = PROVIDER_OPTION,
    model: str | None = MODEL_OPTION,
    light_model: bool = LIGHT_MODEL_OPTION,
    category: ToolCategory = CATEGORY_OPTION,
) -> None:
    """Search対応のLLM呼び出しを実行します."""
    search = resolve_web_search_client(
        provider, category=category, use_light_model=light_model,
    )
    response = search(prompt, model=model)
    typer.echo(response.content)


@tools_app.command("retriever")
def retriever(
    query: str,
    provider: ProviderName = PROVIDER_OPTION,
    model: str | None = MODEL_OPTION,
    top_k: int | None = TOP_K_OPTION,
    threshold: float | None = THRESHOLD_OPTION,
    timeout: float = TIMEOUT_OPTION,
    category: ToolCategory = CATEGORY_OPTION,
) -> None:
    """DB連携の検索リトリーバーを呼び出します."""
    executor = resolve_retriever_client(provider, category=category)
    result = executor(query, model=model, top_k=top_k, threshold=threshold, timeout=timeout)
    typer.echo(json.dumps(result, ensure_ascii=False))
