from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, cast

import typer
from anthropic import Anthropic
from google import genai
from openai import OpenAI
from pydantic import BaseModel, Field

from various_llm_benchmark.interfaces.commands.common import parse_history
from various_llm_benchmark.llm.providers.anthropic.client import AnthropicLLMClient
from various_llm_benchmark.llm.providers.gemini.client import GeminiLLMClient
from various_llm_benchmark.llm.providers.openai.client import OpenAILLMClient
from various_llm_benchmark.models import ChatMessage, LLMResponse
from various_llm_benchmark.prompts.prompt import PromptTemplate, load_provider_prompt
from various_llm_benchmark.settings import settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from various_llm_benchmark.llm.protocol import LLMClient

compare_app = typer.Typer(help="複数プロバイダーの応答を比較します。")

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
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    return GeminiLLMClient(client, settings.gemini_model, temperature=settings.default_temperature)


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

    def run(self) -> list[ComparisonResult]:
        """Execute chat requests for all targets and collect results."""
        results: list[ComparisonResult] = []
        for target in self._targets:
            provider_key = target.normalized_provider
            model_name = target.model or DEFAULT_MODELS.get(provider_key)
            if model_name is None:
                error_message = f"モデルが未設定です: {target.provider}"
                results.append(
                    ComparisonResult(provider=target.provider, model="", error=error_message),
                )
                continue

            factory = self._client_factories.get(provider_key)
            if factory is None:
                error_message = f"未対応のプロバイダーです: {target.provider}"
                results.append(
                    ComparisonResult(provider=target.provider, model=model_name, error=error_message),
                )
                continue

            try:
                response = self._call_provider(provider_key, model_name, factory)
                results.append(
                    ComparisonResult(
                        provider=target.provider,
                        model=response.model,
                        content=response.content,
                    ),
                )
            except Exception as exc:
                results.append(
                    ComparisonResult(
                        provider=target.provider,
                        model=model_name,
                        error=str(exc),
                    ),
                )
        return results

    def _call_provider(
        self,
        provider_key: str,
        model_name: str,
        factory: Callable[[], LLMClient],
    ) -> LLMResponse:
        """Call a provider client using the configured strategy."""
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


def _render_table(results: list[ComparisonResult]) -> str:
    headers = ["Provider", "Model", "Content or Error"]
    rows: list[list[str]] = []
    for result in results:
        body = result.content if result.error is None else f"ERROR: {result.error}"
        normalized_body = body.replace("\n", " ") if body is not None else ""
        rows.append([result.provider, result.model, normalized_body])

    column_widths = [max(len(row[idx]) for row in [headers, *rows]) for idx in range(3)]

    def _format_row(row: list[str]) -> str:
        return " | ".join(value.ljust(column_widths[idx]) for idx, value in enumerate(row))

    divider = "-+-".join("-" * width for width in column_widths)
    formatted_rows = [_format_row(headers), divider]
    formatted_rows.extend(_format_row(row) for row in rows)
    return "\n".join(formatted_rows)


def _results_to_json(results: list[ComparisonResult]) -> str:
    payload = [result.model_dump() for result in results]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _render_results(results: list[ComparisonResult], output_format: str) -> str:
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
) -> None:
    """Run the same prompt across providers and compare responses."""
    messages_history = parse_history(history or [])
    parsed_targets = _parse_targets(targets)
    service = CompareService(prompt=prompt, history=messages_history, targets=parsed_targets)
    results = service.run()

    rendered_output = _render_results(results, output_format)
    typer.echo(rendered_output)

    if output_file is not None:
        output_path = Path(output_file)
        output_path.write_text(_results_to_json(results), encoding="utf-8")
