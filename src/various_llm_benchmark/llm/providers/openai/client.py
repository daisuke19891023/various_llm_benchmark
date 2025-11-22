from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from various_llm_benchmark.llm.protocol import ChatMessage, LLMClient, LLMResponse

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.responses import ResponseInputParam


class OpenAILLMClient(LLMClient):
    """Adapter for OpenAI Responses API."""

    def __init__(self, client: OpenAI, default_model: str, *, temperature: float = 0.7) -> None:
        """Create a client wrapper with defaults."""
        self._client = client
        self._default_model = default_model
        self._temperature = temperature

    def generate(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        """Generate a completion without history."""
        completion = self._client.responses.create(
            model=model or self._default_model,
            input=prompt,
            temperature=self._temperature,
        )
        content = _extract_content(cast("Any", completion.output))
        return LLMResponse(content=content, model=completion.model, raw=completion.model_dump())

    def chat(self, messages: list[ChatMessage], *, model: str | None = None) -> LLMResponse:
        """Generate a completion using chat messages."""
        openai_messages: list[dict[str, str]] = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]
        openai_input = cast("ResponseInputParam", openai_messages)
        completion = self._client.responses.create(
            model=model or self._default_model,
            input=openai_input,
            temperature=self._temperature,
        )
        content = _extract_content(cast("Any", completion.output))
        return LLMResponse(content=content, model=completion.model, raw=completion.model_dump())


def _extract_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_extract_content(item) for item in cast("list[Any]", content))

    text = getattr(content, "text", None)
    if text:
        value = getattr(text, "value", None)
        return value if isinstance(value, str) else str(value)

    nested_content = getattr(content, "content", None)
    if nested_content is not None:
        return _extract_content(nested_content)

    return str(content)
