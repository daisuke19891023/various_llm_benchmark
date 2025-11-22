from __future__ import annotations

from functools import lru_cache

import typer

from various_llm_benchmark.agents.providers import AgnoAgentProvider, ProviderName
from various_llm_benchmark.interfaces.commands.common import build_messages
from various_llm_benchmark.prompts.prompt import PromptTemplate, load_provider_prompt
from various_llm_benchmark.settings import settings

agent_app = typer.Typer(help="Agnoエージェントを呼び出します。")

HISTORY_OPTION: list[str] | None = typer.Option(
    None,
    help="'role:content' 形式の履歴を複数回指定できます。",
    show_default=False,
)

PROVIDER_OPTION = typer.Option(
    "openai",
    "--provider",
    "-p",
    case_sensitive=False,
    help="利用するプロバイダー (openai または anthropic)",
)


@lru_cache(maxsize=1)
def _prompt_template() -> PromptTemplate:
    return load_provider_prompt("agents", "agno")


def _create_provider() -> AgnoAgentProvider:
    prompt_template = _prompt_template()
    return AgnoAgentProvider(
        openai_api_key=settings.openai_api_key.get_secret_value(),
        anthropic_api_key=settings.anthropic_api_key.get_secret_value(),
        openai_model=settings.openai_model,
        anthropic_model=settings.anthropic_model,
        temperature=settings.default_temperature,
        instructions=prompt_template.system,
    )


@agent_app.command("complete")
def agent_complete(
    prompt: str,
    provider: ProviderName = PROVIDER_OPTION,
    model: str | None = typer.Option(None, help="モデル名を上書きします。"),
) -> None:
    """Generate a single-turn response via Agno agent."""
    response = _create_provider().complete(prompt, provider=provider, model=model)
    typer.echo(response.content)


@agent_app.command("chat")
def agent_chat(
    prompt: str,
    history: list[str] | None = HISTORY_OPTION,
    provider: ProviderName = PROVIDER_OPTION,
    model: str | None = typer.Option(None, help="モデル名を上書きします。"),
) -> None:
    """Generate an Agno agent response that includes history."""
    messages = build_messages(prompt, history or [], system_prompt=_prompt_template().system)
    response = _create_provider().chat(messages, provider=provider, model=model)
    typer.echo(response.content)
