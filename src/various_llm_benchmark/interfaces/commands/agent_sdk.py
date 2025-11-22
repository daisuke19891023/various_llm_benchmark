from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import typer

from various_llm_benchmark.agents.providers import OpenAIAgentsProvider
from various_llm_benchmark.interfaces.commands.common import build_messages
from various_llm_benchmark.media.images import read_image_file
from various_llm_benchmark.interfaces.commands.web_search_clients import resolve_web_search_client
from various_llm_benchmark.llm.tools.registry import ToolCategory
from various_llm_benchmark.prompts.prompt import PromptTemplate, load_provider_prompt
from various_llm_benchmark.settings import settings

agent_sdk_app = typer.Typer(help="OpenAI Agents SDK を使って応答します。")

HISTORY_OPTION: list[str] | None = typer.Option(
    None,
    help="'role:content' 形式の履歴を複数回指定できます。",
    show_default=False,
)
LIGHT_MODEL_OPTION: bool = typer.Option(
    default=False,
    help="軽量モデル (gpt-5.1-mini) を使用します。",
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
    return load_provider_prompt("agents", "openai_agents")


def _create_provider(*, use_light_model: bool = False) -> OpenAIAgentsProvider:
    return OpenAIAgentsProvider(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_light_model if use_light_model else settings.openai_model,
        instructions=_prompt_template().system,
        temperature=settings.default_temperature,
    )


@agent_sdk_app.command("complete")
def agent_sdk_complete(prompt: str, light_model: bool = LIGHT_MODEL_OPTION) -> None:
    """OpenAI Agents SDK による単発応答を実行します."""
    response = _create_provider(use_light_model=light_model).complete(prompt)
    typer.echo(response.content)


@agent_sdk_app.command("chat")
def agent_sdk_chat(
    prompt: str, history: list[str] | None = HISTORY_OPTION, light_model: bool = LIGHT_MODEL_OPTION,
) -> None:
    """OpenAI Agents SDK による履歴付き応答を実行します."""
    messages = build_messages(prompt, history or [], system_prompt=_prompt_template().system)
    response = _create_provider(use_light_model=light_model).chat(messages)
    typer.echo(response.content)


@agent_sdk_app.command("web-search")
def agent_sdk_web_search(
    prompt: str,
    model: str | None = typer.Option(default=None, help="モデル名を上書きします。"),
    light_model: bool = LIGHT_MODEL_OPTION,
    category: ToolCategory = CATEGORY_OPTION,
) -> None:
    """OpenAIの組み込みWeb検索ツールをAgents SDK経由で呼び出します."""
    search = resolve_web_search_client(
        "openai", category=category, use_light_model=light_model,
    )
    response = search(prompt, model=model)
    typer.echo(response.content)


@agent_sdk_app.command("vision")
def agent_sdk_vision(
    prompt: str,
    image_path: Path = IMAGE_ARGUMENT,
    light_model: bool = LIGHT_MODEL_OPTION,
) -> None:
    """OpenAI Agents SDK による画像付き推論を実行します."""
    resolved_path = Path(image_path)
    image_input = read_image_file(resolved_path)
    response = _create_provider(use_light_model=light_model).vision(prompt, image_input)
    typer.echo(response.content)
