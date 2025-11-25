from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import typer
from rich.console import Console

if TYPE_CHECKING:
    from various_llm_benchmark.agents.providers import AgnoAgentProvider
    from various_llm_benchmark.agents.providers.agno import ProviderName


from various_llm_benchmark.interfaces.commands.common import build_messages
from various_llm_benchmark.interfaces.commands.retriever_clients import resolve_retriever_client
from various_llm_benchmark.interfaces.commands.web_search_clients import resolve_web_search_client
from various_llm_benchmark.llm.tools.registry import ToolCategory
from various_llm_benchmark.prompts.prompt import PromptTemplate, load_provider_prompt

agent_app = typer.Typer(help="Agnoエージェントを呼び出します。")
console = Console()

RetrieverProviderName = Literal["openai", "google", "voyage"]

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

PROVIDER_OPTION = typer.Option(
    "openai",
    "--provider",
    "-p",
    case_sensitive=False,
    help="利用するプロバイダー (openai / anthropic / gemini)",
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


@lru_cache(maxsize=1)
def _prompt_template() -> PromptTemplate:
    return load_provider_prompt("agents", "agno")


def _create_provider(*, use_light_model: bool = False) -> AgnoAgentProvider:
    from various_llm_benchmark.agents.providers import AgnoAgentProvider
    from various_llm_benchmark.settings import settings

    prompt_template = _prompt_template()
    openai_model = settings.openai_light_model if use_light_model else settings.openai_model
    anthropic_model = settings.anthropic_light_model if use_light_model else settings.anthropic_model
    gemini_model = settings.gemini_light_model if use_light_model else settings.gemini_model
    return AgnoAgentProvider(
        openai_api_key=settings.openai_api_key.get_secret_value(),
        anthropic_api_key=settings.anthropic_api_key.get_secret_value(),
        gemini_api_key=settings.gemini_api_key.get_secret_value(),
        openai_model=openai_model,
        anthropic_model=anthropic_model,
        gemini_model=gemini_model,
        temperature=settings.default_temperature,
        instructions=prompt_template.system,
    )


@agent_app.command("complete")
def agent_complete(
    prompt: str,
    provider: str = PROVIDER_OPTION,
    model: str | None = MODEL_OPTION,
    light_model: bool = LIGHT_MODEL_OPTION,
) -> None:
    """Generate a single-turn response via Agno agent."""
    provider_client = _create_provider(use_light_model=light_model)
    with console.status("Agnoエージェントで応答生成中...", spinner="dots"):
        response = provider_client.complete(prompt, provider=cast("ProviderName", provider), model=model)
    console.print(response.content)


@agent_app.command("chat")
def agent_chat(
    prompt: str,
    history: list[str] | None = HISTORY_OPTION,
    provider: str = PROVIDER_OPTION,
    model: str | None = MODEL_OPTION,
    light_model: bool = LIGHT_MODEL_OPTION,
) -> None:
    """Generate an Agno agent response that includes history."""
    messages = build_messages(prompt, history or [], system_prompt=_prompt_template().system)
    provider_client = _create_provider(use_light_model=light_model)
    with console.status("Agnoエージェントで履歴付き応答を生成中...", spinner="dots"):
        response = provider_client.chat(messages, provider=cast("ProviderName", provider), model=model)
    console.print(response.content)


@agent_app.command("web-search")
def agent_web_search(
    prompt: str,
    provider: str = PROVIDER_OPTION,
    model: str | None = MODEL_OPTION,
    light_model: bool = LIGHT_MODEL_OPTION,
    category: ToolCategory = CATEGORY_OPTION,
) -> None:
    """組み込みWeb検索ツールをAgno経由で呼び出します."""
    search = resolve_web_search_client(
        cast("ProviderName", provider),
        category=category,
        use_light_model=light_model,
    )
    with console.status("Web検索ツールを呼び出し中...", spinner="dots"):
        response = search(prompt, model=model)
    console.print(response.content)


@agent_app.command("retriever")
def agent_retriever(
    query: str,
    provider: RetrieverProviderName = RETRIEVER_PROVIDER_OPTION,
    model: str | None = MODEL_OPTION,
    top_k: int | None = TOP_K_OPTION,
    threshold: float | None = THRESHOLD_OPTION,
    timeout: float = TIMEOUT_OPTION,
    category: ToolCategory = CATEGORY_OPTION,
) -> None:
    """DB連携のリトリーバーツールをAgno経由で呼び出します."""
    retriever = resolve_retriever_client(provider, category=category)
    with console.status("リトリーバーツールを実行中...", spinner="dots"):
        result = retriever(
            query,
            model=model,
            top_k=top_k,
            threshold=threshold,
            timeout=timeout,
        )
    console.print_json(json.dumps(result, ensure_ascii=False))


@agent_app.command("vision")
def agent_vision(
    prompt: str,
    image_path: Path = IMAGE_ARGUMENT,
    provider: str = PROVIDER_OPTION,
    model: str | None = MODEL_OPTION,
    light_model: bool = LIGHT_MODEL_OPTION,
) -> None:
    """Agnoエージェントで画像入力を含めた応答を生成します."""
    from various_llm_benchmark.media.images import read_image_file

    resolved_path = Path(image_path)
    image_input = read_image_file(resolved_path)
    provider_client = _create_provider(use_light_model=light_model)
    with console.status("画像を含む応答を生成中...", spinner="dots"):
        response = provider_client.vision(prompt, image_input, provider=cast("ProviderName", provider), model=model)
    console.print(response.content)
