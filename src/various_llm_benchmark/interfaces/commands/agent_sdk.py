from __future__ import annotations

from functools import lru_cache

import typer

from various_llm_benchmark.agents.providers import OpenAIAgentsProvider
from various_llm_benchmark.interfaces.commands.common import build_messages
from various_llm_benchmark.prompts.prompt import PromptTemplate, load_provider_prompt
from various_llm_benchmark.settings import settings

agent_sdk_app = typer.Typer(help="OpenAI Agents SDK を使って応答します。")

HISTORY_OPTION: list[str] | None = typer.Option(
    None,
    help="'role:content' 形式の履歴を複数回指定できます。",
    show_default=False,
)


@lru_cache(maxsize=1)
def _prompt_template() -> PromptTemplate:
    return load_provider_prompt("agents", "openai_agents")


def _create_provider() -> OpenAIAgentsProvider:
    return OpenAIAgentsProvider(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_model,
        instructions=_prompt_template().system,
        temperature=settings.default_temperature,
    )


@agent_sdk_app.command("complete")
def agent_sdk_complete(prompt: str) -> None:
    """OpenAI Agents SDK による単発応答を実行します."""
    response = _create_provider().complete(prompt)
    typer.echo(response.content)


@agent_sdk_app.command("chat")
def agent_sdk_chat(prompt: str, history: list[str] | None = HISTORY_OPTION) -> None:
    """OpenAI Agents SDK による履歴付き応答を実行します."""
    messages = build_messages(prompt, history or [], system_prompt=_prompt_template().system)
    response = _create_provider().chat(messages)
    typer.echo(response.content)
