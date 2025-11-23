from __future__ import annotations

import asyncio
import json
from functools import cache
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, cast
from textwrap import shorten

import typer
from anthropic import Anthropic
from google.genai import Client
from openai import OpenAI
from pydantic import BaseModel, Field
from rich.box import SIMPLE_HEAD
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table

from various_llm_benchmark.interfaces.commands.common import parse_history
from various_llm_benchmark.interfaces.runner import (
    AsyncJobRunner,
    TaskHooks,
    TaskResult,
)
from various_llm_benchmark.llm.providers.anthropic.client import AnthropicLLMClient
from various_llm_benchmark.llm.providers.gemini.client import GeminiLLMClient
from various_llm_benchmark.llm.providers.openai.client import OpenAILLMClient
from various_llm_benchmark.models import ChatMessage, LLMResponse
from various_llm_benchmark.prompts.prompt import PromptTemplate, load_provider_prompt
from various_llm_benchmark.settings import settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from various_llm_benchmark.llm.protocol import LLMClient

compare_app = typer.Typer(help="複数プロバイダーの応答をProgress表示付きで比較します。")
console = Console()

HISTORY_OPTION: list[str] | None = typer.Option(
    None,
    help="'role:content' 形式の履歴を複数回指定できます。",
    show_default=False,
)
TARGET_OPTION: list[str] | None = typer.Option(
    None,
    "--target",
    "-t",
    help=(
        "比較対象のプロバイダーとモデルを 'provider:model' 形式で指定します。"
        "モデル省略時はデフォルト設定を使用します。"
    ),
)
OUTPUT_FORMAT_OPTION: str = typer.Option(
    "table",
    "--format",
    "-f",
    case_sensitive=False,
    help="出力形式 (table/json)。",
)
OUTPUT_FILE_OPTION: Path | None = typer.Option(
    None,
    "--output-file",
    "-o",
    help="結果をJSON形式で保存するファイルパス。",
    writable=True,
)
CONCURRENCY_OPTION: int = typer.Option(
    3,
    "--concurrency",
    "-c",
    min=1,
    help="同時に実行するタスク数。",
)
RETRY_OPTION: int = typer.Option(
    0,
    "--retries",
    "-r",
    min=0,
    help="失敗時に再試行する回数。",
)

PROVIDER_ALIASES: dict[str, str] = {"claude": "anthropic"}
DEFAULT_MODELS: dict[str, str] = {
    "openai": settings.openai_model,
    "anthropic": settings.anthropic_model,
    "gemini": settings.gemini_model,
}
PROMPT_KEYS: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
}


class ComparisonTarget(BaseModel):
    """Target provider and model pair."""

    provider: str = Field(..., description="プロバイダー名")
    model: str | None = Field(None, description="利用するモデル名")

    @property
    def normalized_provider(self) -> str:
        """Return provider key normalized for lookup."""
        return PROVIDER_ALIASES.get(self.provider.lower(), self.provider.lower())


class ComparisonResult(BaseModel):
    """Standardized result for comparison output."""

    provider: str
    model: str
    content: str | None = None
    error: str | None = None


@cache
def _prompt_template(provider_key: str) -> PromptTemplate:
    prompt_key = PROMPT_KEYS.get(provider_key, provider_key)
    return load_provider_prompt("llm", prompt_key)


def _openai_client() -> OpenAILLMClient:
    client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
    return OpenAILLMClient(client, settings.openai_model, temperature=settings.default_temperature)


def _anthropic_client() -> AnthropicLLMClient:
    client = Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
    return AnthropicLLMClient(client, settings.anthropic_model, temperature=settings.default_temperature)


def _gemini_client() -> GeminiLLMClient:
    client = Client(api_key=settings.gemini_api_key.get_secret_value())
    return GeminiLLMClient(
        client,
        settings.gemini_model,
        temperature=settings.default_temperature,
        thinking_level=settings.gemini_thinking_level,
    )


CLIENT_FACTORIES: dict[str, Callable[[], LLMClient]] = {
    "openai": _openai_client,
    "anthropic": _anthropic_client,
    "gemini": _gemini_client,
}


class CompareService:
    """Orchestrates chat calls across providers."""

    def __init__(
        self,
        *,
        prompt: str,
        history: list[ChatMessage],
        targets: list[ComparisonTarget],
        client_factories: dict[str, Callable[[], LLMClient]] | None = None,
    ) -> None:
        """Create a service for running comparisons."""
        self._prompt = prompt
        self._history = history
        self._targets = targets
        self._client_factories = client_factories or CLIENT_FACTORIES

    async def run_async(
        self,
        *,
        concurrency: int = 3,
        max_retries: int = 0,
        hooks: TaskHooks | None = None,
    ) -> list[ComparisonResult]:
        """Execute chat requests for all targets and collect results asynchronously."""
        runner: AsyncJobRunner[ComparisonResult] = AsyncJobRunner(
            concurrency=concurrency,
            max_retries=max_retries,
            hooks=hooks,
        )
        for target in self._targets:
            runner.add_task(self._execute_target, target, name=f"compare-{target.provider}")

        task_results: list[TaskResult[ComparisonResult]] = await runner.run()
        results: list[ComparisonResult] = []
        for task_result, target in zip(task_results, self._targets, strict=False):
            if task_result.cancelled:
                results.append(
                    ComparisonResult(
                        provider=target.provider,
                        model=target.model or "",
                        error="タスクがキャンセルされました",
                    ),
                )
                continue

            if task_result.error is not None:
                results.append(
                    ComparisonResult(
                        provider=target.provider,
                        model=target.model or "",
                        error=str(task_result.error),
                    ),
                )
                continue

            if task_result.result is None:
                error_message = "結果が設定されていません"
                results.append(
                    ComparisonResult(
                        provider=target.provider,
                        model=target.model or "",
                        error=error_message,
                    ),
                )
                continue

            results.append(task_result.result)

        return results

    def run(
        self,
        *,
        concurrency: int = 3,
        max_retries: int = 0,
        hooks: TaskHooks | None = None,
    ) -> list[ComparisonResult]:
        """Run comparisons synchronously via :meth:`run_async`."""
        return asyncio.run(
            self.run_async(concurrency=concurrency, max_retries=max_retries, hooks=hooks),
        )

    async def _execute_target(self, target: ComparisonTarget) -> ComparisonResult:
        provider_key = target.normalized_provider
        model_name = target.model or DEFAULT_MODELS.get(provider_key)
        if model_name is None:
            error_message = f"モデルが未設定です: {target.provider}"
            raise ValueError(error_message)

        factory = self._client_factories.get(provider_key)
        if factory is None:
            error_message = f"未対応のプロバイダーです: {target.provider}"
            raise ValueError(error_message)

        response = await self._call_provider(provider_key, model_name, factory)
        return ComparisonResult(
            provider=target.provider,
            model=response.model,
            content=response.content,
        )

    async def _call_provider(
        self,
        provider_key: str,
        model_name: str,
        factory: Callable[[], LLMClient],
    ) -> LLMResponse:
        """Call a provider client using the configured strategy in a worker thread."""
        return await asyncio.to_thread(
            self._call_provider_sync,
            provider_key,
            model_name,
            factory,
        )

    def _call_provider_sync(
        self,
        provider_key: str,
        model_name: str,
        factory: Callable[[], LLMClient],
    ) -> LLMResponse:
        client = factory()
        if provider_key == "gemini":
            gemini_client = cast("GeminiLLMClient", client)
            messages = list(self._history)
            messages.append(ChatMessage(role="user", content=self._prompt))
            template = _prompt_template(provider_key)
            return gemini_client.chat(
                messages,
                model=model_name,
                system_instruction=template.system,
            )

        template = _prompt_template(provider_key)
        messages = template.to_messages(self._prompt, self._history)
        typed_client: LLMClient = client
        return typed_client.chat(messages, model=model_name)


def _short_error(message: str) -> str:
    return shorten(message, width=80, placeholder="…")


def _build_progress(
    progress: Progress,
    targets: list[ComparisonTarget],
) -> dict[str, TaskID]:
    task_ids: dict[str, TaskID] = {}
    for target in targets:
        provider_key = target.normalized_provider
        model_label = target.model or DEFAULT_MODELS.get(provider_key, "default")
        description = f"{target.provider}:{model_label}"
        task_name = f"compare-{target.provider}"
        task_ids[task_name] = progress.add_task(
            description,
            total=1,
            start=False,
            fields={"status": "待機中"},
        )
    return task_ids


class ProgressTaskHooks(TaskHooks):
    """Render task lifecycle updates to a :class:`Progress` instance."""

    def __init__(self, progress: Progress, task_ids: dict[str, TaskID]) -> None:
        """Initialize with a progress renderer and registered task IDs."""
        self._progress = progress
        self._task_ids = task_ids

    def _task_id(self, name: str) -> TaskID | None:
        return self._task_ids.get(name)

    def on_start(self, name: str, attempt: int) -> None:
        """Start a task and mark it as running."""
        task_id = self._task_id(name)
        if task_id is None:
            return
        self._progress.start_task(task_id)
        self._progress.update(
            task_id,
            completed=0,
            fields={"status": f"[cyan]実行中[/cyan] (試行 {attempt})"},
        )

    def on_retry(self, name: str, attempt: int, error: BaseException) -> None:
        """Show retry status with the latest error message."""
        task_id = self._task_id(name)
        if task_id is None:
            return
        message = _short_error(str(error))
        self._progress.update(
            task_id,
            fields={"status": f"[yellow]リトライ[/yellow] (試行 {attempt}, 失敗: {message})"},
        )

    def on_success(self, name: str, attempt: int) -> None:
        """Mark a task as successfully finished."""
        task_id = self._task_id(name)
        if task_id is None:
            return
        self._progress.update(
            task_id,
            completed=1,
            fields={"status": f"[green]完了[/green] ({attempt}回目)"},
        )

    def on_failure(self, name: str, attempt: int, error: BaseException) -> None:
        """Surface a terminal failure with the latest attempt count."""
        task_id = self._task_id(name)
        if task_id is None:
            return
        message = _short_error(str(error))
        self._progress.update(
            task_id,
            completed=1,
            fields={"status": f"[red]失敗[/red] (試行 {attempt}): {message}"},
        )

    def on_cancel(self, name: str, attempt: int) -> None:
        """Mark a task as cancelled."""
        task_id = self._task_id(name)
        if task_id is None:
            return
        self._progress.update(
            task_id,
            completed=1,
            fields={"status": f"[magenta]キャンセル[/magenta] (試行 {attempt})"},
        )


def _parse_targets(targets: list[str] | None) -> list[ComparisonTarget]:
    if not targets:
        return [
            ComparisonTarget(provider="openai", model=settings.openai_model),
            ComparisonTarget(provider="claude", model=settings.anthropic_model),
            ComparisonTarget(provider="gemini", model=settings.gemini_model),
        ]

    parsed_targets: list[ComparisonTarget] = []
    for target in targets:
        if ":" in target:
            provider, model = target.split(":", 1)
            parsed_targets.append(ComparisonTarget(provider=provider, model=model))
        else:
            parsed_targets.append(ComparisonTarget(provider=target, model=None))
    return parsed_targets


def _render_table(results: list[ComparisonResult]) -> Table:
    table = Table(title="Comparison Results", box=SIMPLE_HEAD)
    table.add_column("Provider", style="bold")
    table.add_column("Model")
    table.add_column("Content or Error", overflow="fold")

    for result in results:
        body = (
            f"[red]ERROR[/red]: {result.error}"
            if result.error is not None
            else result.content or ""
        )
        table.add_row(result.provider, result.model, body)
    return table


def _results_to_json(results: list[ComparisonResult]) -> str:
    payload = [result.model_dump() for result in results]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _render_results(results: list[ComparisonResult], output_format: str) -> Table | str:
    normalized_format = output_format.lower()
    if normalized_format == "table":
        return _render_table(results)
    if normalized_format == "json":
        return _results_to_json(results)
    error_message = "format は 'table' または 'json' で指定してください。"
    raise typer.BadParameter(error_message)


@compare_app.command("chat")
def compare_chat(
    prompt: str,
    history: list[str] | None = HISTORY_OPTION,
    targets: list[str] | None = TARGET_OPTION,
    output_format: str = OUTPUT_FORMAT_OPTION,
    output_file: Path | None = OUTPUT_FILE_OPTION,
    concurrency: int = CONCURRENCY_OPTION,
    retries: int = RETRY_OPTION,
) -> None:
    """Run the same prompt across providers and compare responses."""
    messages_history = parse_history(history or [])
    parsed_targets = _parse_targets(targets)
    service = CompareService(prompt=prompt, history=messages_history, targets=parsed_targets)
    console.print(
        f"[bold cyan]比較を開始します[/bold cyan] "
        f"(タスク {len(parsed_targets)} 件, 並列 {concurrency}, リトライ {retries})",
    )
    start_time = perf_counter()
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TextColumn("{task.fields[status]}", justify="right"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_ids = _build_progress(progress, parsed_targets)
        hooks = ProgressTaskHooks(progress, task_ids)
        results = asyncio.run(
            service.run_async(
                concurrency=concurrency,
                max_retries=retries,
                hooks=hooks,
            ),
        )

    elapsed_seconds = perf_counter() - start_time
    console.print(f"[green]比較完了[/green] (経過 {elapsed_seconds:.2f} 秒)")

    rendered_output = _render_results(results, output_format)
    if isinstance(rendered_output, Table):
        console.print(rendered_output)
    else:
        console.print_json(rendered_output)

    if output_file is not None:
        output_path = Path(output_file)
        with console.status(f"結果を {output_path} に保存中...", spinner="dots"):
            output_path.write_text(_results_to_json(results), encoding="utf-8")
        console.print(f"[green]保存完了[/green]: {output_path}")
