"""Utilities for constructing web search tool callers with shared defaults."""

from __future__ import annotations

from functools import lru_cache, partial
from typing import TYPE_CHECKING, Literal, Protocol, cast

from anthropic import Anthropic
from google import genai
from openai import OpenAI

from various_llm_benchmark.llm.tools.web_search import (
    AnthropicWebSearchTool,
    GeminiWebSearchTool,
    OpenAIWebSearchTool,
    SupportsMessages,
    SupportsResponses,
    SupportsSearchModels,
)
from various_llm_benchmark.llm.tools.registry import (
    ToolCategory,
    ToolRegistration,
    get_tool,
    register_tool,
)
from various_llm_benchmark.prompts.prompt import PromptTemplate, load_provider_prompt
from various_llm_benchmark.settings import settings

if TYPE_CHECKING:
    from various_llm_benchmark.models import LLMResponse

ProviderName = Literal["openai", "anthropic", "gemini"]


class WebSearchHandler(Protocol):
    """Callable that executes a web search-enabled LLM request."""

    def __call__(
        self,
        prompt: str,
        *,
        model: str | None = None,
        use_light_model: bool = False,
    ) -> LLMResponse:
        """Execute a search request."""
        ...


class WebSearchExecutor(Protocol):
    """Callable that runs a web search for a prompt."""

    def __call__(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        """Execute a search request."""
        ...

WEB_SEARCH_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string", "description": "ユーザーからの検索依頼"},
        "model": {"type": "string", "description": "明示的に利用するモデル"},
        "use_light_model": {
            "type": "boolean",
            "description": "軽量モデルを利用する場合にtrue",
        },
    },
    "required": ["prompt"],
}


@lru_cache(maxsize=1)
def _openai_prompt_template() -> PromptTemplate:
    return load_provider_prompt("llm", "openai")


@lru_cache(maxsize=1)
def _anthropic_prompt_template() -> PromptTemplate:
    return load_provider_prompt("llm", "anthropic")


@lru_cache(maxsize=1)
def _gemini_prompt_template() -> PromptTemplate:
    return load_provider_prompt("llm", "gemini")


@lru_cache(maxsize=2)
def build_openai_web_search_tool(use_light_model: bool = False) -> OpenAIWebSearchTool:
    """Return a cached OpenAI web search caller."""
    client = cast("SupportsResponses", OpenAI(api_key=settings.openai_api_key.get_secret_value()))
    default_model = settings.openai_light_model if use_light_model else settings.openai_model
    return OpenAIWebSearchTool(
        client,
        default_model,
        temperature=settings.default_temperature,
        system_prompt=_openai_prompt_template().system,
    )


@lru_cache(maxsize=2)
def build_anthropic_web_search_tool(use_light_model: bool = False) -> AnthropicWebSearchTool:
    """Return a cached Anthropic web search caller."""
    client = cast("SupportsMessages", Anthropic(api_key=settings.anthropic_api_key.get_secret_value()))
    default_model = settings.anthropic_light_model if use_light_model else settings.anthropic_model
    return AnthropicWebSearchTool(
        client,
        default_model,
        temperature=settings.default_temperature,
        system_prompt=_anthropic_prompt_template().system,
    )


@lru_cache(maxsize=2)
def build_gemini_web_search_tool(use_light_model: bool = False) -> GeminiWebSearchTool:
    """Return a cached Gemini web search caller."""
    client = cast("SupportsSearchModels", genai.Client(api_key=settings.gemini_api_key.get_secret_value()))
    default_model = settings.gemini_light_model if use_light_model else settings.gemini_model
    return GeminiWebSearchTool(
        client,
        default_model,
        temperature=settings.default_temperature,
        system_prompt=_gemini_prompt_template().system,
    )


def _tool_id(provider: ProviderName) -> str:
    return f"web-search/{provider}"


def _register_web_search_tool(
    provider: ProviderName,
    description: str,
    handler: WebSearchHandler,
) -> None:
    register_tool(
        ToolRegistration(
            id=_tool_id(provider),
            description=description,
            input_schema=WEB_SEARCH_INPUT_SCHEMA,
            handler=handler,
            category=ToolCategory.BUILTIN,
        ),
    )


def _openai_web_search(
    prompt: str, *, model: str | None = None, use_light_model: bool = False,
) -> LLMResponse:
    tool = build_openai_web_search_tool(use_light_model=use_light_model)
    return tool.search(prompt, model=model)


def _anthropic_web_search(
    prompt: str, *, model: str | None = None, use_light_model: bool = False,
) -> LLMResponse:
    tool = build_anthropic_web_search_tool(use_light_model=use_light_model)
    return tool.search(prompt, model=model)


def _gemini_web_search(
    prompt: str, *, model: str | None = None, use_light_model: bool = False,
) -> LLMResponse:
    tool = build_gemini_web_search_tool(use_light_model=use_light_model)
    return tool.search(prompt, model=model)


def _ensure_web_search_tools_registered() -> None:
    """Register built-in web search tool adapters."""
    for provider, description, handler in (
        (
            "openai",
            "OpenAI Responses API を使ったビルトインWeb検索ツール",
            _openai_web_search,
        ),
        (
            "anthropic",
            "Anthropic Messages API を使ったビルトインWeb検索ツール",
            _anthropic_web_search,
        ),
        (
            "gemini",
            "Gemini Search API を使ったビルトインWeb検索ツール",
            _gemini_web_search,
        ),
    ):
        try:
            _register_web_search_tool(
                cast("ProviderName", provider),
                description,
                handler,
            )
        except ValueError:
            # Tools are registered at import time; ignore duplicate registrations.
            continue


def resolve_web_search_client(
    provider: ProviderName,
    *,
    category: ToolCategory = ToolCategory.BUILTIN,
    use_light_model: bool = False,
) -> WebSearchExecutor:
    """Construct a callable search executor from the registry."""
    _ensure_web_search_tools_registered()
    registration = get_tool(_tool_id(provider), category=category)
    handler = cast("WebSearchHandler", registration.handler)
    return cast("WebSearchExecutor", partial(handler, use_light_model=use_light_model))
