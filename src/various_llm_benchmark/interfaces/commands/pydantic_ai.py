from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import typer
from rich.console import Console

from various_llm_benchmark.agents.providers import PydanticAIAgentProvider
from various_llm_benchmark.interfaces.commands.common import build_messages
from various_llm_benchmark.interfaces.commands.web_search_clients import resolve_web_search_client
from various_llm_benchmark.llm.tools.registry import ToolCategory
from various_llm_benchmark.media.images import read_image_file
from various_llm_benchmark.prompts.prompt import PromptTemplate, load_provider_prompt
from various_llm_benchmark.settings import settings

pydantic_ai_app = typer.Typer(help="Pydantic AIエージェントを呼び出します。")
console = Console()

HISTORY_OPTION: list[str] | None = typer.Option(
    None,
    help="'role:content' 形式の履歴を複数回指定できます。",
    show_default=False,
)
MODEL_OPTION: str | None = typer.Option(default=None, help="モデル名を上書きします。")
LIGHT_MODEL_OPTION: bool = typer.Option(
    default=False,
    help="軽量モデル (gpt-5.1-mini) を使用します。",
)
TEMPERATURE_OPTION: float | None = typer.Option(
    default=None,
    min=0.0,
    max=2.0,
    help="温度を指定します。未指定の場合は設定値を使用します。",
)
CATEGORY_OPTION: ToolCategory = typer.Option(
    default=ToolCategory.BUILTIN,
    case_sensitive=False,
    help="利用するツールカテゴリ (builtin / mcp / external)",
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
    return load_provider_prompt("agents", "pydantic_ai")


def create_provider(
    *,
    model: str | None = None,
    use_light_model: bool = False,
    temperature: float | None = None,
) -> PydanticAIAgentProvider:
    """Create a configured Pydantic AI provider instance."""
    pydantic_ai_key = settings.pydantic_ai_api_key.get_secret_value()
    openai_key = settings.openai_api_key.get_secret_value()
    if pydantic_ai_key:
        os.environ.setdefault("PYDANTIC_AI_API_KEY", pydantic_ai_key)
    if openai_key:
        os.environ.setdefault("OPENAI_API_KEY", openai_key)

    resolved_model = model or (
        settings.pydantic_ai_light_model
        if use_light_model
        else settings.pydantic_ai_model
    )
    resolved_temperature = settings.default_temperature if temperature is None else temperature
    return PydanticAIAgentProvider(
        model=resolved_model,
        system_prompt=_prompt_template().system,
        temperature=resolved_temperature,
    )


@pydantic_ai_app.command("complete")
def pydantic_ai_complete(
    prompt: str,
    model: str | None = MODEL_OPTION,
    light_model: bool = LIGHT_MODEL_OPTION,
    temperature: float | None = TEMPERATURE_OPTION,
) -> None:
    """Pydantic AIで単発応答を生成します."""
    provider = create_provider(
        model=model, use_light_model=light_model, temperature=temperature,
    )
    with console.status("Pydantic AIで応答生成中...", spinner="dots"):
        response = provider.complete(prompt)
    console.print(response.content)


@pydantic_ai_app.command("chat")
def pydantic_ai_chat(
    prompt: str,
    history: list[str] | None = HISTORY_OPTION,
    model: str | None = MODEL_OPTION,
    light_model: bool = LIGHT_MODEL_OPTION,
    temperature: float | None = TEMPERATURE_OPTION,
) -> None:
    """履歴付きでPydantic AIの応答を生成します."""
    messages = build_messages(prompt, history or [], system_prompt=_prompt_template().system)
    provider = create_provider(
        model=model, use_light_model=light_model, temperature=temperature,
    )
    with console.status("Pydantic AIで履歴付き応答を生成中...", spinner="dots"):
        response = provider.chat(messages)
    console.print(response.content)


@pydantic_ai_app.command("vision")
def pydantic_ai_vision(
    prompt: str,
    image_path: Path = IMAGE_ARGUMENT,
    model: str | None = MODEL_OPTION,
    light_model: bool = LIGHT_MODEL_OPTION,
    temperature: float | None = TEMPERATURE_OPTION,
) -> None:
    """Pydantic AIで画像入力を含めた応答を生成します."""
    resolved_path = Path(image_path)
    image_input = read_image_file(resolved_path)
    provider = create_provider(
        model=model, use_light_model=light_model, temperature=temperature,
    )
    with console.status("Pydantic AIで画像を解析中...", spinner="dots"):
        response = provider.vision(prompt, image_input)
    console.print(response.content)


@pydantic_ai_app.command("web-search")
def pydantic_ai_web_search(
    prompt: str,
    model: str | None = MODEL_OPTION,
    light_model: bool = LIGHT_MODEL_OPTION,
    category: ToolCategory = CATEGORY_OPTION,
) -> None:
    """OpenAI Web検索ツールをPydantic AI CLIから呼び出します."""
    search = resolve_web_search_client(
        "openai", category=category, use_light_model=light_model,
    )
    with console.status("Pydantic AI経由でWeb検索中...", spinner="dots"):
        response = search(prompt, model=model)
    console.print(response.content)
