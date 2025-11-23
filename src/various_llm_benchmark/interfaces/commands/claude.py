from __future__ import annotations

from functools import lru_cache
from pathlib import Path


import typer
from rich.console import Console
from anthropic import Anthropic

from various_llm_benchmark.interfaces.commands.common import build_messages
from various_llm_benchmark.media.images import read_image_file
from various_llm_benchmark.llm.providers.anthropic.client import AnthropicLLMClient
from various_llm_benchmark.prompts.prompt import PromptTemplate, load_provider_prompt
from various_llm_benchmark.settings import settings

claude_app = typer.Typer(help="Claudeモデルを呼び出します。")
console = Console()

HISTORY_OPTION: list[str] | None = typer.Option(
    None,
    help="'role:content' 形式の履歴を複数回指定できます。",
    show_default=False,
)
IMAGE_ARGUMENT = typer.Argument(
    ...,
    exists=True,
    readable=True,
    dir_okay=False,
    help="解析する画像ファイルのパス",
)


@lru_cache(maxsize=1)
def _prompt_template() -> PromptTemplate:
    return load_provider_prompt("llm", "anthropic")


def _client() -> AnthropicLLMClient:
    client = Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
    return AnthropicLLMClient(client, settings.anthropic_model, temperature=settings.default_temperature)


def _thinking_config(enabled: bool, budget_tokens: int) -> dict[str, object] | None:
    if not enabled:
        return None
    return {"type": "enabled", "budget_tokens": budget_tokens}


@claude_app.command("complete")
def claude_complete(
    prompt: str,
    model: str | None = typer.Option(None, help="モデル名を上書きします."),
    extended_thinking: bool = typer.Option(
        default=False,
        help="Claude Extended Thinkingを有効化します。",
    ),
    thinking_tokens: int = typer.Option(
        default=8192,
        help="Extended Thinkingに割り当てるトークン数。",
    ),
) -> None:
    """Generate a single-turn response with Claude."""
    with console.status("Claudeで応答生成中...", spinner="dots"):
        response = _client().generate(
            _prompt_template().to_prompt_text(prompt),
            model=model,
            thinking=_thinking_config(extended_thinking, thinking_tokens),
        )
    console.print(response.content)


@claude_app.command("chat")
def claude_chat(
    prompt: str,
    history: list[str] | None = HISTORY_OPTION,
    model: str | None = typer.Option(None, help="モデル名を上書きします。"),
    extended_thinking: bool = typer.Option(
        default=False,
        help="Claude Extended Thinkingを有効化します。",
    ),
    thinking_tokens: int = typer.Option(
        default=8192,
        help="Extended Thinkingに割り当てるトークン数。",
    ),
) -> None:
    """Generate a chat response with optional history."""
    messages = build_messages(prompt, history or [], system_prompt=_prompt_template().system)
    with console.status("Claudeで履歴付き応答を生成中...", spinner="dots"):
        response = _client().chat(
            messages,
            model=model,
            thinking=_thinking_config(extended_thinking, thinking_tokens),
        )
    console.print(response.content)


@claude_app.command("vision")
def claude_vision(
    prompt: str,
    image_path: Path = IMAGE_ARGUMENT,
    model: str | None = typer.Option(None, help="モデル名を上書きします。"),
) -> None:
    """Analyze an image with a Claude model."""
    resolved_path = Path(image_path)
    image_input = read_image_file(resolved_path)
    with console.status("Claudeで画像を解析中...", spinner="dots"):
        response = _client().vision(
            prompt,
            image_input,
            model=model,
            system_prompt=_prompt_template().system,
        )
    console.print(response.content)
