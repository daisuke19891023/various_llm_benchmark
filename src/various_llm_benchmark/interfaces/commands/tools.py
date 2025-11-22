from __future__ import annotations

from typing import Literal

import typer

from various_llm_benchmark.interfaces.commands.web_search_clients import resolve_web_search_client

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


@tools_app.command("web-search")
def web_search(
    prompt: str,
    provider: ProviderName = PROVIDER_OPTION,
    model: str | None = MODEL_OPTION,
    light_model: bool = LIGHT_MODEL_OPTION,
) -> None:
    """Search対応のLLM呼び出しを実行します."""
    client = resolve_web_search_client(provider, use_light_model=light_model)
    response = client.search(prompt, model=model)
    typer.echo(response.content)
