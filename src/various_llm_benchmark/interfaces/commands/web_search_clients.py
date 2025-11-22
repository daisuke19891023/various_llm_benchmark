"""Utilities for constructing web search tool callers with shared defaults."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, cast

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
from various_llm_benchmark.prompts.prompt import PromptTemplate, load_provider_prompt
from various_llm_benchmark.settings import settings

ProviderName = Literal["openai", "anthropic", "gemini"]


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


def resolve_web_search_client(
    provider: ProviderName, *, use_light_model: bool = False,
) -> AnthropicWebSearchTool | OpenAIWebSearchTool | GeminiWebSearchTool:
    """Construct a search-enabled client for the given provider."""
    if provider == "anthropic":
        return build_anthropic_web_search_tool(use_light_model=use_light_model)
    if provider == "gemini":
        return build_gemini_web_search_tool(use_light_model=use_light_model)
    return build_openai_web_search_tool(use_light_model=use_light_model)
