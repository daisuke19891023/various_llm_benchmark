from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from various_llm_benchmark.llm.protocol import LLMClient
from various_llm_benchmark.models import ChatMessage, LLMResponse

if TYPE_CHECKING:
    from anthropic import Anthropic
    from anthropic.types import MessageParam


class AnthropicLLMClient(LLMClient):
    """Adapter for Anthropic Messages API."""

    def __init__(self, client: Anthropic, default_model: str, *, temperature: float = 0.7) -> None:
        """Create a client wrapper with defaults."""
        self._client = client
        self._default_model = default_model
        self._temperature = temperature

    def generate(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        """Generate a completion without history."""
        messages = cast("list[MessageParam]", [{"role": "user", "content": prompt}])
        response = self._client.messages.create(
            model=model or self._default_model,
            messages=messages,
            max_tokens=1024,
            temperature=self._temperature,
        )
        content_data = cast("list[Any] | str", response.content)
        content = _extract_text(content_data)
        return LLMResponse(content=content, model=response.model, raw=response.model_dump())

    def chat(self, messages: list[ChatMessage], *, model: str | None = None) -> LLMResponse:
        """Generate a completion using chat messages."""
        anthropic_messages = cast(
            "list[MessageParam]",
            [{"role": msg.role, "content": msg.content} for msg in messages],
        )
        response = self._client.messages.create(
            model=model or self._default_model,
            messages=anthropic_messages,
            max_tokens=1024,
            temperature=self._temperature,
        )
        content_data = cast("list[Any] | str", response.content)
        content = _extract_text(content_data)
        return LLMResponse(content=content, model=response.model, raw=response.model_dump())


def _extract_text(content: list[Any] | str) -> str:
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if isinstance(block, Mapping):
            mapping_block = cast("Mapping[str, Any]", block)
            text_value = mapping_block.get("text") or mapping_block.get("content")
            if text_value is not None:
                parts.append(str(text_value))
                continue
            parts.append(str(mapping_block))
            continue
        text_attr = cast("Any", getattr(block, "text", None))
        if text_attr is not None:
            parts.append(str(text_attr))
            continue
        parts.append(str(block))
    return "".join(parts)


def extract_anthropic_text(content: list[Any] | str) -> str:
    """Public wrapper for parsing Anthropic message content."""
    return _extract_text(content)
