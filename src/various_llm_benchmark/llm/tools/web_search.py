from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

from various_llm_benchmark.llm.providers.anthropic.client import extract_anthropic_text
from various_llm_benchmark.llm.providers.gemini.client import extract_gemini_text
from various_llm_benchmark.llm.providers.openai.client import extract_openai_content
from various_llm_benchmark.models import LLMResponse

if TYPE_CHECKING:
    from anthropic.types import MessageParam
    from openai.types.responses import ResponseInputParam


class OpenAIResponse(Protocol):
    """Subset of fields returned from the OpenAI Responses API."""

    model: str
    output: Any

    def model_dump(self) -> dict[str, object]:
        """Return a raw mapping of the response."""
        ...


class ResponsesClient(Protocol):
    """Minimal interface for the OpenAI responses client."""

    def create(self, **kwargs: object) -> OpenAIResponse:
        """Create a response request."""
        ...


class SupportsResponses(Protocol):
    """Protocol for OpenAI clients exposing the responses API."""

    responses: ResponsesClient


class AnthropicResponse(Protocol):
    """Subset of fields returned from the Anthropic Messages API."""

    model: str
    content: list[Any] | str

    def model_dump(self) -> dict[str, object]:
        """Return a raw mapping of the response."""
        ...


class MessagesClient(Protocol):
    """Minimal interface for the Anthropic messages client."""

    def create(self, **kwargs: object) -> AnthropicResponse:
        """Create a messages request."""
        ...


class SupportsMessages(Protocol):
    """Protocol for Anthropic clients exposing the messages API."""

    messages: MessagesClient


class GeminiModelsClient(Protocol):
    """Minimal interface for Gemini model operations."""

    def generate_content(self, **kwargs: object) -> object:
        """Call a Gemini model."""
        ...


class SupportsSearchModels(Protocol):
    """Protocol for Gemini clients exposing model calls."""

    models: GeminiModelsClient


class OpenAIWebSearchTool:
    """Call OpenAI's built-in web search tool via the Responses API."""

    def __init__(
        self,
        client: SupportsResponses,
        default_model: str,
        *,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> None:
        """Configure the tool caller."""
        self._client = client
        self._default_model = default_model
        self._temperature = temperature
        self._system_prompt = system_prompt

    def search(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        """Execute a search-enabled completion."""
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.responses.create(
            model=model or self._default_model,
            input=cast("ResponseInputParam", messages),
            temperature=self._temperature,
            tools=[{"type": "web_search"}],
            tool_choice="auto",
        )
        content = extract_openai_content(response.output)
        return LLMResponse(content=content, model=response.model, raw=response.model_dump())


class AnthropicWebSearchTool:
    """Call Claude's built-in web search tool via the Messages API."""

    def __init__(
        self,
        client: SupportsMessages,
        default_model: str,
        *,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> None:
        """Configure the tool caller."""
        self._client = client
        self._default_model = default_model
        self._temperature = temperature
        self._system_prompt = system_prompt

    def search(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        """Execute a Claude search request."""
        messages = cast("list[MessageParam]", [{"role": "user", "content": prompt}])
        request_kwargs: dict[str, object] = {
            "model": model or self._default_model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": self._temperature,
            "tools": [{"type": "web_search"}],
            "tool_choice": {"type": "auto"},
        }
        if self._system_prompt is not None:
            request_kwargs["system"] = self._system_prompt

        response = self._client.messages.create(**request_kwargs)
        content = extract_anthropic_text(response.content)
        return LLMResponse(content=content, model=response.model, raw=response.model_dump())


class GeminiWebSearchTool:
    """Call Gemini's web search tool via the Generative Models API."""

    def __init__(
        self,
        client: SupportsSearchModels,
        default_model: str,
        *,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> None:
        """Configure the Gemini tool caller."""
        self._client = client
        self._default_model = default_model
        self._temperature = temperature
        self._system_prompt = system_prompt

    def search(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        """Execute a Gemini search request."""
        contents = cast("list[dict[str, object]]", [{"role": "user", "parts": [prompt]}])
        request_kwargs: dict[str, object] = {
            "model": model or self._default_model,
            "contents": contents,
            "config": {"temperature": self._temperature},
            "tools": [{"google_search_retrieval": {}}],
            "tool_config": {
                "google_search_retrieval": {"dynamic_retrieval_config": {"mode": "DYNAMIC"}},
            },
        }
        if self._system_prompt is not None:
            request_kwargs["system_instruction"] = self._system_prompt

        models_client = cast("Any", self._client.models)
        response = models_client.generate_content(**request_kwargs)
        content = extract_gemini_text(response)
        response_model = getattr(response, "model", None) or request_kwargs["model"]
        raw_output = getattr(response, "model_dump", lambda: {"data": response})()
        return LLMResponse(content=content, model=cast("str", response_model), raw=raw_output)
