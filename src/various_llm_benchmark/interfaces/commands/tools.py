from __future__ import annotations

import json
from typing import Literal

import typer
from rich.console import Console

from various_llm_benchmark.interfaces.commands.code_execution_clients import (
    resolve_code_execution_client,
)
from various_llm_benchmark.interfaces.commands.retriever_clients import resolve_retriever_client
from various_llm_benchmark.interfaces.commands.shell_clients import resolve_shell_client
from various_llm_benchmark.interfaces.commands.web_search_clients import resolve_web_search_client
from various_llm_benchmark.llm.tools.registry import ToolCategory

ProviderName = Literal["openai", "anthropic", "gemini"]
RetrieverProviderName = Literal["openai", "google", "voyage"]
ShellProviderName = Literal["openai", "anthropic"]


tools_app = typer.Typer(help="LLMの組み込みツール呼び出しを実行します。")
console = Console()

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
    help="シェルコマンドのタイムアウト (秒)",
    min=0.01,
)
SHELL_ARGS_OPTION: list[str] = typer.Option(
    default_factory=list,
    help="シェルコマンドに渡す追加引数 (スペース区切りで指定)",
    show_default=False,
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
    console.print_json(result.model_dump_json(ensure_ascii=False))


@tools_app.command("code-execution")
def code_execution(
    prompt: str,
    provider: ProviderName = PROVIDER_OPTION,
    model: str | None = MODEL_OPTION,
    light_model: bool = LIGHT_MODEL_OPTION,
) -> None:
    """ネイティブなコード実行ツールを有効化したLLM呼び出しを行います.

    ``llm/tools/code_execution.py`` で登録しているサンドボックス実行は、Agent経由の
    ツール呼び出し用の安全なローカル実行を担います。このサブコマンドでは各プロ
    バイダーが提供するネイティブコード実行機能の疎通を直接確認します。
    """
    executor = resolve_code_execution_client(provider, use_light_model=light_model)
    with console.status("コード実行ツールを呼び出し中...", spinner="dots"):
        response = executor(prompt, model=model)
    console.print(response.content)


@tools_app.command("shell")
def shell_execution(
    command: str,
    args: list[str] = SHELL_ARGS_OPTION,
    provider: ShellProviderName = SHELL_PROVIDER_OPTION,
    timeout_seconds: float = SHELL_TIMEOUT_OPTION,
    category: ToolCategory = CATEGORY_OPTION,
) -> None:
    """許可されたシェルコマンドをツール経由で実行します."""
    executor = resolve_shell_client(provider, category=category)
    with console.status("シェルコマンドを実行中...", spinner="dots"):
        result = executor(command, args=args, timeout_seconds=timeout_seconds)
    console.print_json(json.dumps(result, ensure_ascii=False))
