from __future__ import annotations

from functools import lru_cache

import typer

from various_llm_benchmark.interfaces.commands.common import build_messages
from various_llm_benchmark.llm.providers.dspy import DsPyLLMClient
from various_llm_benchmark.prompts.prompt import PromptTemplate, load_provider_prompt
from various_llm_benchmark.settings import settings


dspy_app = typer.Typer(help="DsPyモデルを呼び出します。")

HISTORY_OPTION: list[str] | None = typer.Option(
    None,
    help="'role:content' 形式の履歴を複数回指定できます。",
    show_default=False,
)


@lru_cache(maxsize=1)
def _prompt_template() -> PromptTemplate:
    return load_provider_prompt("llm", "dspy")


def _client() -> DsPyLLMClient:
    return DsPyLLMClient(
        settings.openai_model,
        temperature=settings.default_temperature,
        api_key=settings.openai_api_key.get_secret_value(),
    )


def _selected_model(model: str | None) -> str | None:
    if isinstance(model, typer.models.OptionInfo):
        return None
    return model


@dspy_app.command("complete")
def dspy_complete(prompt: str, model: str | None = typer.Option(None, help="モデル名を上書きします。")) -> None:
    """Generate a single-turn response with DsPy."""
    response = _client().generate(
        _prompt_template().to_prompt_text(prompt), model=_selected_model(model),
    )
    typer.echo(response.content)


@dspy_app.command("chat")
def dspy_chat(
    prompt: str,
    history: list[str] | None = HISTORY_OPTION,
    model: str | None = typer.Option(None, help="モデル名を上書きします。"),
) -> None:
    """Generate a chat response with optional history."""
    messages = build_messages(prompt, history or [], system_prompt=_prompt_template().system)
    response = _client().chat(messages, model=_selected_model(model))
    typer.echo(response.content)
