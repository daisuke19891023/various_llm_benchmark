from __future__ import annotations

import typer
from anthropic import Anthropic

from various_llm_benchmark.interfaces.commands.common import build_messages
from various_llm_benchmark.llm.providers.anthropic.client import AnthropicLLMClient
from various_llm_benchmark.settings import settings

claude_app = typer.Typer(help="Claudeモデルを呼び出します。")

HISTORY_OPTION: list[str] | None = typer.Option(
    None,
    help="'role:content' 形式の履歴を複数回指定できます。",
    show_default=False,
)


def _client() -> AnthropicLLMClient:
    client = Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
    return AnthropicLLMClient(client, settings.anthropic_model, temperature=settings.default_temperature)


@claude_app.command("complete")
def claude_complete(prompt: str, model: str | None = typer.Option(None, help="モデル名を上書きします.")) -> None:
    """Generate a single-turn response with Claude."""
    response = _client().generate(prompt, model=model)
    typer.echo(response.content)


@claude_app.command("chat")
def claude_chat(
    prompt: str,
    history: list[str] | None = HISTORY_OPTION,
    model: str | None = typer.Option(None, help="モデル名を上書きします。"),
) -> None:
    """Generate a chat response with optional history."""
    messages = build_messages(prompt, history or [])
    response = _client().chat(messages, model=model)
    typer.echo(response.content)
