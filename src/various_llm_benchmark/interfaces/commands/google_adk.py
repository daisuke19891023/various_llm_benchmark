from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import typer

from various_llm_benchmark.agents.providers import GoogleADKProvider
from various_llm_benchmark.interfaces.commands.common import build_messages
from various_llm_benchmark.interfaces.commands.web_search_clients import build_gemini_web_search_tool
from various_llm_benchmark.media.images import read_image_file
from various_llm_benchmark.prompts.prompt import PromptTemplate, load_provider_prompt
from various_llm_benchmark.settings import settings

adk_app = typer.Typer(help="Google ADK エージェントを呼び出します。")

HISTORY_OPTION: list[str] | None = typer.Option(
    None,
    help="'role:content' 形式の履歴を複数回指定できます。",
    show_default=False,
)
LIGHT_MODEL_OPTION: bool = typer.Option(
    default=False,
    help="軽量モデル (gemini-2.5-flash) を使用します。",
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
    return load_provider_prompt("agents", "google_adk")


def _create_provider(*, use_light_model: bool = False) -> GoogleADKProvider:
    return GoogleADKProvider(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.gemini_light_model if use_light_model else settings.gemini_model,
        instructions=_prompt_template().system,
        temperature=settings.default_temperature,
    )


@adk_app.command("complete")
def adk_complete(prompt: str, light_model: bool = LIGHT_MODEL_OPTION) -> None:
    """Google ADKによる単発応答を実行します."""
    response = _create_provider(use_light_model=light_model).complete(prompt)
    typer.echo(response.content)


@adk_app.command("chat")
def adk_chat(prompt: str, history: list[str] | None = HISTORY_OPTION, light_model: bool = LIGHT_MODEL_OPTION) -> None:
    """Google ADKによる履歴付き応答を実行します."""
    messages = build_messages(prompt, history or [], system_prompt=_prompt_template().system)
    response = _create_provider(use_light_model=light_model).chat(messages)
    typer.echo(response.content)


@adk_app.command("web-search")
def adk_web_search(
    prompt: str,
    model: str | None = typer.Option(default=None, help="モデル名を上書きします。"),
    light_model: bool = LIGHT_MODEL_OPTION,
) -> None:
    """GeminiのWeb検索ツールをADKと同じキーで呼び出します."""
    client = build_gemini_web_search_tool(use_light_model=light_model)
    response = client.search(prompt, model=model)
    typer.echo(response.content)


@adk_app.command("vision")
def adk_vision(prompt: str, image_path: Path = IMAGE_ARGUMENT, light_model: bool = LIGHT_MODEL_OPTION) -> None:
    """Google ADKによる画像付き推論を実行します."""
    resolved_path = Path(image_path)
    image_input = read_image_file(resolved_path)
    response = _create_provider(use_light_model=light_model).vision(prompt, image_input)
    typer.echo(response.content)
