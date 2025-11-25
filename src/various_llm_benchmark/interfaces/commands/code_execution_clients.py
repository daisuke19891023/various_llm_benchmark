"""Provider-specific callers for native code execution tools."""
from __future__ import annotations

from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from various_llm_benchmark.llm.providers.anthropic.client import extract_anthropic_text
from various_llm_benchmark.llm.providers.gemini.client import extract_gemini_text
from various_llm_benchmark.llm.providers.openai.client import extract_openai_content
from various_llm_benchmark.models import LLMResponse, normalize_tool_calls
from various_llm_benchmark.prompts.prompt import PromptTemplate, load_provider_prompt
from various_llm_benchmark.settings import settings

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

    class SupportsModels(Protocol):
        """Protocol for Gemini clients exposing model calls."""

        models: GeminiModelsClient

ProviderName = Literal["openai", "anthropic", "gemini"]


class CodeExecutionExecutor(Protocol):
    """Callable that executes a code-enabled LLM request."""

    def __call__(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        """Execute a code-enabled completion."""
        ...


class OpenAICodeExecutionTool:
    """Call OpenAI's built-in code interpreter via the Responses API."""

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

    def run(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        """Execute a completion that can run code."""
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.responses.create(
            model=model or self._default_model,
            input=cast("ResponseInputParam", messages),
            temperature=self._temperature,
            tools=[{"type": "code_interpreter"}],
            tool_choice="auto",
        )
        raw = response.model_dump()
        content = extract_openai_content(response.output)
        return LLMResponse(
            content=content,
            model=response.model,
            raw=raw,
            tool_calls=normalize_tool_calls(raw),
        )


class AnthropicCodeExecutionTool:
    """Call Claude's code execution tool via the Messages API."""

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

    def run(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        """Execute a completion that can run code."""
        messages = cast("list[MessageParam]", [{"role": "user", "content": prompt}])
        request_kwargs: dict[str, object] = {
            "model": model or self._default_model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": self._temperature,
            "tools": [{"type": "code_execution_20250825", "name": "code_execution"}],
            "tool_choice": {"type": "auto"},
        }
        if self._system_prompt is not None:
            request_kwargs["system"] = self._system_prompt

        response = self._client.messages.create(**request_kwargs)
        raw = response.model_dump()
        content = extract_anthropic_text(response.content)
        return LLMResponse(
            content=content,
            model=response.model,
            raw=raw,
            tool_calls=normalize_tool_calls(raw),
        )


class GeminiCodeExecutionTool:
    """Call Gemini's code execution tool via the Generative Models API."""

    def __init__(
        self,
        client: SupportsModels,
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

    def run(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        """Execute a completion that can run code."""
        contents = cast("list[dict[str, object]]", [{"role": "user", "parts": [prompt]}])
        request_kwargs: dict[str, object] = {
            "model": model or self._default_model,
            "contents": contents,
            "config": {"temperature": self._temperature},
            "tools": [{"code_execution": {}}],
            "tool_config": {"code_execution": {"mode": "AUTO"}},
        }
        if self._system_prompt is not None:
            request_kwargs["system_instruction"] = self._system_prompt

        response = self._client.models.generate_content(**request_kwargs)
        content = extract_gemini_text(response)
        response_model = getattr(response, "model", None) or request_kwargs["model"]
        raw_output = getattr(response, "model_dump", lambda: {"data": response})()
        raw = cast("dict[str, object]", raw_output) if isinstance(raw_output, dict) else {"data": raw_output}
        return LLMResponse(
            content=content,
            model=cast("str", response_model),
            raw=raw,
            tool_calls=normalize_tool_calls(raw),
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
def build_openai_code_execution_tool(use_light_model: bool = False) -> OpenAICodeExecutionTool:
    """Return a cached OpenAI code execution caller."""
    from openai import OpenAI

    client = cast("SupportsResponses", OpenAI(api_key=settings.openai_api_key.get_secret_value()))
    default_model = settings.openai_light_model if use_light_model else settings.openai_model
    return OpenAICodeExecutionTool(
        client,
        default_model,
        temperature=settings.default_temperature,
        system_prompt=_openai_prompt_template().system,
    )


@lru_cache(maxsize=2)
def build_anthropic_code_execution_tool(use_light_model: bool = False) -> AnthropicCodeExecutionTool:
    """Return a cached Anthropic code execution caller."""
    from anthropic import Anthropic

    client = cast("SupportsMessages", Anthropic(api_key=settings.anthropic_api_key.get_secret_value()))
    default_model = settings.anthropic_light_model if use_light_model else settings.anthropic_model
    return AnthropicCodeExecutionTool(
        client,
        default_model,
        temperature=settings.default_temperature,
        system_prompt=_anthropic_prompt_template().system,
    )


@lru_cache(maxsize=2)
def build_gemini_code_execution_tool(use_light_model: bool = False) -> GeminiCodeExecutionTool:
    """Return a cached Gemini code execution caller."""
    from google.genai import Client

    client = cast("SupportsModels", Client(api_key=settings.gemini_api_key.get_secret_value()))
    default_model = settings.gemini_light_model if use_light_model else settings.gemini_model
    return GeminiCodeExecutionTool(
        client,
        default_model,
        temperature=settings.default_temperature,
        system_prompt=_gemini_prompt_template().system,
    )


def resolve_code_execution_client(
    provider: ProviderName,
    *,
    use_light_model: bool = False,
) -> CodeExecutionExecutor:
    """Construct a callable code execution executor for the given provider."""
    builders = {
        "openai": build_openai_code_execution_tool,
        "anthropic": build_anthropic_code_execution_tool,
        "gemini": build_gemini_code_execution_tool,
    }
    try:
        builder = builders[provider]
    except KeyError as exc:
        msg = f"Unsupported provider: {provider}"
        raise ValueError(msg) from exc
    tool = builder(use_light_model)
    return cast("CodeExecutionExecutor", partial(tool.run))
