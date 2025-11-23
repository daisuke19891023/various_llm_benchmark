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
LIGHT_MODEL_OPTION: bool = typer.Option(
    default=False,
    help="軽量モデルを使用します。",
    show_default=False,
)


@lru_cache(maxsize=1)
def _prompt_template() -> PromptTemplate:
    return load_provider_prompt("llm", "dspy")


def _client() -> DsPyLLMClient:
    default_model = settings.dspy_model
    return DsPyLLMClient(
        default_model,
        temperature=settings.default_temperature,
    )


def _resolve_model(model: str | None, light_model: bool) -> str | None:
    if model is not None:
        return model
    if light_model:
        return settings.dspy_light_model
    return None


@dspy_app.command("complete")
def dspy_complete(
    prompt: str,
    model: str | None = typer.Option(None, help="モデル名を上書きします。"),
    light_model: bool = LIGHT_MODEL_OPTION,
) -> None:
    """Generate a single-turn response with DsPy."""
    response = _client().generate(
        _prompt_template().to_prompt_text(prompt),
        model=_resolve_model(model, light_model),
    )
    typer.echo(response.content)


@dspy_app.command("chat")
def dspy_chat(
    prompt: str,
    history: list[str] | None = HISTORY_OPTION,
    model: str | None = typer.Option(None, help="モデル名を上書きします。"),
    light_model: bool = LIGHT_MODEL_OPTION,
) -> None:
    """Generate a chat response with optional history."""
    messages = build_messages(prompt, history or [], system_prompt=_prompt_template().system)
    response = _client().chat(messages, model=_resolve_model(model, light_model))
    typer.echo(response.content)
