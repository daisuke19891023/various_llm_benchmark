from __future__ import annotations

import json
from typing import Literal

import typer
from rich.console import Console

from various_llm_benchmark.interfaces.commands.retriever_clients import resolve_retriever_client
from various_llm_benchmark.interfaces.commands.shell_clients import (
    ShellProviderName,
    ShellResult,
    resolve_shell_client,
)
from various_llm_benchmark.interfaces.commands.web_search_clients import resolve_web_search_client
from various_llm_benchmark.llm.tools.registry import ToolCategory

ProviderName = Literal["openai", "anthropic", "gemini"]
RetrieverProviderName = Literal["openai", "google", "voyage"]


tools_app = typer.Typer(help="LLMの組み込みツール呼び出しを実行します。")
console = Console()

ARG_OPTION: list[str] | None = typer.Option(
    None,
    "--arg",
    "-a",
    help="シェルコマンドに渡す引数。複数指定できます。",
    show_default=False,
)

PROVIDER_OPTION: ProviderName = typer.Option(
    "openai",
    "--provider",
    "-p",
    case_sensitive=False,
    help="利用するプロバイダー (openai / anthropic / gemini)",
)
SHELL_PROVIDER_OPTION: ShellProviderName = typer.Option(
    "openai",
    "--provider",
    "-p",
    case_sensitive=False,
    help="シェルツールで利用するプロバイダー (openai / anthropic)",
)

RETRIEVER_PROVIDER_OPTION: RetrieverProviderName = typer.Option(
    "openai",
    "--provider",
    "-p",
    case_sensitive=False,
    help="リトリーバーで利用する埋め込みプロバイダー (openai / google / voyage)",
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
SHELL_TIMEOUT_OPTION: float = typer.Option(
    5.0,
    "--timeout-seconds",
    help="シェルコマンド実行のタイムアウト (秒)",
    min=0.01,
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
        provider,
        category=category,
        use_light_model=light_model,
    )
    with console.status("組み込みWeb検索ツールを呼び出し中...", spinner="dots"):
        response = search(prompt, model=model)
    console.print(response.content)


@tools_app.command("retriever")
def retriever(
    query: str,
    provider: RetrieverProviderName = RETRIEVER_PROVIDER_OPTION,
    model: str | None = MODEL_OPTION,
    top_k: int | None = TOP_K_OPTION,
    threshold: float | None = THRESHOLD_OPTION,
    timeout: float = TIMEOUT_OPTION,
    category: ToolCategory = CATEGORY_OPTION,
) -> None:
    """DB連携の検索リトリーバーを呼び出します."""
    executor = resolve_retriever_client(provider, category=category)
    with console.status("リトリーバーを実行中...", spinner="dots"):
        result = executor(
            query,
            model=model,
            top_k=top_k,
            threshold=threshold,
            timeout=timeout,
        )
    console.print_json(json.dumps(result, ensure_ascii=False))


@tools_app.command("shell")
def shell(
    command: str,
    args: list[str] | None = ARG_OPTION,
    provider: ShellProviderName = SHELL_PROVIDER_OPTION,
    timeout_seconds: float = SHELL_TIMEOUT_OPTION,
) -> None:
    """許可されたシェルコマンドを組み込みツール経由で実行します."""
    executor = resolve_shell_client(provider)
    with console.status(f"{provider}のシェルツールを実行中...", spinner="dots"):
        result: ShellResult = executor(command, args=args, timeout_seconds=timeout_seconds)

    stdout = result["stdout"]
    stderr = result["stderr"]
    output = stdout or stderr
    console.print(output)
