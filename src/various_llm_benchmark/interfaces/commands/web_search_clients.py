"""Utilities for constructing web search tool callers with shared defaults."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Literal, Protocol, cast, overload

if TYPE_CHECKING:
    from various_llm_benchmark.llm.tools.web_search import (
        AnthropicWebSearchTool,
        GeminiWebSearchTool,
        OpenAIWebSearchTool,
    )


from various_llm_benchmark.llm.tools import ToolSelector
from various_llm_benchmark.llm.tools.registry import (
    WEB_SEARCH_TAG,
    NativeToolType,
    ToolCategory,
    ToolRegistration,
    register_tool,
)
from various_llm_benchmark.llm.tools.types import WebSearchInput
from various_llm_benchmark.prompts.prompt import PromptTemplate, load_provider_prompt
from various_llm_benchmark.settings import settings

if TYPE_CHECKING:
    from various_llm_benchmark.models import LLMResponse

ProviderName = Literal["openai", "anthropic", "gemini"]


class WebSearchHandler(Protocol):
    """Callable that executes a web search-enabled LLM request."""

    @overload
    def __call__(
        self,
        request: WebSearchInput,
        *,
        model: str | None = None,
        use_light_model: bool = False,
    ) -> LLMResponse:
        ...

    @overload
    def __call__(
        self,
        request: str,
        *,
        model: str | None = None,
        use_light_model: bool = False,
    ) -> LLMResponse:
        ...

    def __call__(self, *args: object, **kwargs: object) -> LLMResponse:
        """Execute a search request."""
        ...


class WebSearchExecutor(Protocol):
    """Callable that runs a web search for a prompt."""

    @overload
    def __call__(
        self,
        request: WebSearchInput,
        *,
        model: str | None = None,
        use_light_model: bool | None = None,
    ) -> LLMResponse:
        ...

    @overload
    def __call__(
        self,
        request: str,
        *,
        model: str | None = None,
        use_light_model: bool | None = None,
    ) -> LLMResponse:
        ...

    def __call__(self, *args: object, **kwargs: object) -> LLMResponse:
        """Execute a search request."""
        ...


WEB_SEARCH_INPUT_SCHEMA: dict[str, object] = cast(
    "dict[str, object]", WebSearchInput.model_json_schema(),
)


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
    from openai import OpenAI

    from various_llm_benchmark.llm.tools.web_search import OpenAIWebSearchTool, SupportsResponses

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
    from anthropic import Anthropic

    from various_llm_benchmark.llm.tools.web_search import AnthropicWebSearchTool, SupportsMessages

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
    from google.genai import Client

    from various_llm_benchmark.llm.tools.web_search import GeminiWebSearchTool, SupportsSearchModels

    client = cast("SupportsSearchModels", Client(api_key=settings.gemini_api_key.get_secret_value()))
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
            name=f"{provider}-web-search",
            description=description,
            input_schema=WEB_SEARCH_INPUT_SCHEMA,
            input_model=WebSearchInput,
            tags={WEB_SEARCH_TAG, f"provider:{provider}"},
            native_type=NativeToolType.WEB_SEARCH,
            handler=handler,
            category=ToolCategory.BUILTIN,
        ),
    )


def _normalize_web_search_input(
    request: str | WebSearchInput,
    *,
    model: str | None,
    use_light_model: bool,
) -> tuple[str, str | None, bool]:
    if isinstance(request, WebSearchInput):
        resolved_model = model if model is not None else request.model
        resolved_use_light_model = (
            request.use_light_model
            if request.use_light_model is not None
            else use_light_model
        )
        return request.prompt, resolved_model, bool(resolved_use_light_model)
    return request, model, use_light_model


def _openai_web_search(
    request: str | WebSearchInput,
    *,
    model: str | None = None,
    use_light_model: bool = False,
) -> LLMResponse:
    prompt, resolved_model, resolved_use_light_model = _normalize_web_search_input(
        request,
        model=model,
        use_light_model=use_light_model,
    )
    tool = build_openai_web_search_tool(use_light_model=resolved_use_light_model)
    return tool.search(prompt, model=resolved_model)


def _anthropic_web_search(
    request: str | WebSearchInput,
    *,
    model: str | None = None,
    use_light_model: bool = False,
) -> LLMResponse:
    prompt, resolved_model, resolved_use_light_model = _normalize_web_search_input(
        request,
        model=model,
        use_light_model=use_light_model,
    )
    tool = build_anthropic_web_search_tool(use_light_model=resolved_use_light_model)
    return tool.search(prompt, model=resolved_model)


def _gemini_web_search(
    request: str | WebSearchInput,
    *,
    model: str | None = None,
    use_light_model: bool = False,
) -> LLMResponse:
    prompt, resolved_model, resolved_use_light_model = _normalize_web_search_input(
        request,
        model=model,
        use_light_model=use_light_model,
    )
    tool = build_gemini_web_search_tool(use_light_model=resolved_use_light_model)
    return tool.search(prompt, model=resolved_model)


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
    selector = ToolSelector()
    registration = selector.select_one(
        category=category,
        names=[f"{provider}-web-search"],
        tags=[WEB_SEARCH_TAG, f"provider:{provider}"],
        ids=[_tool_id(provider)],
    )
    handler = cast("WebSearchHandler", registration.handler)

    def executor(
        request: str | WebSearchInput,
        *,
        model: str | None = None,
        use_light_model_override: bool | None = None,
    ) -> LLMResponse:
        resolved_use_light_model = (
            use_light_model_override
            if use_light_model_override is not None
            else use_light_model
        )
        return handler(request, model=model, use_light_model=resolved_use_light_model)

    return cast("WebSearchExecutor", executor)
