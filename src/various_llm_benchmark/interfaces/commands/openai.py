from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import typer
from openai import OpenAI

from various_llm_benchmark.interfaces.commands.common import build_messages
from various_llm_benchmark.media.images import read_image_file
from various_llm_benchmark.llm.providers.openai.client import OpenAILLMClient
from various_llm_benchmark.prompts.prompt import PromptTemplate, load_provider_prompt
from various_llm_benchmark.settings import settings

openai_app = typer.Typer(help="OpenAIモデルを呼び出します。")

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
    return load_provider_prompt("llm", "openai")


def _client() -> OpenAILLMClient:
    client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
    return OpenAILLMClient(client, settings.openai_model, temperature=settings.default_temperature)


@openai_app.command("complete")
def openai_complete(prompt: str, model: str | None = typer.Option(None, help="モデル名を上書きします。")) -> None:
    """Generate a single-turn response with OpenAI."""
    response = _client().generate(_prompt_template().to_prompt_text(prompt), model=model)
    typer.echo(response.content)


@openai_app.command("chat")
def openai_chat(
    prompt: str,
    history: list[str] | None = HISTORY_OPTION,
    model: str | None = typer.Option(None, help="モデル名を上書きします。"),
) -> None:
    """Generate a chat response with optional history."""
    messages = build_messages(prompt, history or [], system_prompt=_prompt_template().system)
    response = _client().chat(messages, model=model)
    typer.echo(response.content)


@openai_app.command("vision")
def openai_vision(
    prompt: str,
    image_path: Path = IMAGE_ARGUMENT,
    model: str | None = typer.Option(None, help="モデル名を上書きします。"),
) -> None:
    """Analyze an image with an OpenAI model."""
    resolved_path = Path(image_path)
    image_input = read_image_file(resolved_path)
    response = _client().vision(
        prompt,
        image_input,
        model=model,
        system_prompt=_prompt_template().system,
    )
    typer.echo(response.content)
