from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import typer
from rich.console import Console

from various_llm_benchmark.interfaces.commands.common import build_messages
from various_llm_benchmark.llm.providers.dspy import DsPyLLMClient
from various_llm_benchmark.llm.providers.dspy.optimizer import (
    PromptOptimizationResult,
    load_prompt_tuning_examples,
    optimize_prompt,
)
from various_llm_benchmark.prompts.prompt import PromptTemplate, load_provider_prompt
from various_llm_benchmark.settings import settings


dspy_app = typer.Typer(help="DsPyモデルを呼び出します。")
console = Console()

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


def _resolve_model_or_default(model: str | None, light_model: bool) -> str:
    return _resolve_model(model, light_model) or settings.dspy_model


@dspy_app.command("complete")
def dspy_complete(
    prompt: str,
    model: str | None = typer.Option(None, help="モデル名を上書きします。"),
    light_model: bool = LIGHT_MODEL_OPTION,
) -> None:
    """Generate a single-turn response with DsPy."""
    with console.status("DsPyで応答生成中...", spinner="dots"):
        response = _client().generate(
            _prompt_template().to_prompt_text(prompt),
            model=_resolve_model(model, light_model),
        )
    console.print(response.content)


@dspy_app.command("chat")
def dspy_chat(
    prompt: str,
    history: list[str] | None = HISTORY_OPTION,
    model: str | None = typer.Option(None, help="モデル名を上書きします。"),
    light_model: bool = LIGHT_MODEL_OPTION,
) -> None:
    """Generate a chat response with optional history."""
    messages = build_messages(prompt, history or [], system_prompt=_prompt_template().system)
    with console.status("DsPyで履歴付き応答を生成中...", spinner="dots"):
        response = _client().chat(messages, model=_resolve_model(model, light_model))
    console.print(response.content)


@dspy_app.command("optimize")
def dspy_optimize(
    dataset_path: Path,
    max_bootstrapped_demos: int = typer.Option(4, help="初期デモ数の上限"),
    num_candidates: int = typer.Option(8, help="探索する候補数"),
    num_threads: int = typer.Option(1, help="探索時のスレッド数"),
    model: str | None = typer.Option(None, help="モデル名を上書きします。"),
    light_model: bool = LIGHT_MODEL_OPTION,
) -> None:
    """DsPy Optimizerを使ってプロンプトをチューニングします."""
    dataset = Path(dataset_path)
    examples = load_prompt_tuning_examples(dataset)
    with console.status("DsPyでプロンプトを最適化中...", spinner="dots"):
        result: PromptOptimizationResult = optimize_prompt(
            examples,
            _prompt_template(),
            model=_resolve_model_or_default(model, light_model),
            temperature=settings.default_temperature,
            max_bootstrapped_demos=max_bootstrapped_demos,
            num_candidates=num_candidates,
            num_threads=num_threads,
        )

    console.print("[bold]=== DsPy Prompt Optimization ===[/bold]")
    console.print(f"データセット件数: {result.trainset_size}")
    console.print(f"ベーススコア: {result.base_score:.3f}")
    console.print(f"最適化後スコア: {result.optimized_score:.3f}")
