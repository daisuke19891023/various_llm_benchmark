from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from various_llm_benchmark.llm.protocol import LLMClient
from various_llm_benchmark.logger import BaseComponent
from various_llm_benchmark.models import (
    ChatMessage,
    ImageInput,
    LLMResponse,
    normalize_tool_calls,
)

if TYPE_CHECKING:
    from anthropic import Anthropic
    from anthropic.types import MessageParam


class AnthropicLLMClient(LLMClient, BaseComponent):
    """Adapter for Anthropic Messages API."""

    def __init__(self, client: Anthropic, default_model: str, *, temperature: float = 0.7) -> None:
        """Create a client wrapper with defaults."""
        self._client = client
        self._default_model = default_model
        self._temperature = temperature

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        thinking: Mapping[str, object] | None = None,
    ) -> LLMResponse:
        """Generate a completion without history."""
        resolved_model = model or self._default_model
        self.log_start("anthropic_generate", model=resolved_model)
        self.log_io(direction="input", prompt=prompt)
        messages = cast("list[MessageParam]", [{"role": "user", "content": prompt}])
        request_kwargs = _build_messages_kwargs(
            resolved_model,
            messages,
            temperature=self._temperature,
            thinking=thinking,
        )
        messages_client = cast("Any", self._client.messages)
        start = perf_counter()
        response = messages_client.create(**request_kwargs)
        elapsed_seconds = perf_counter() - start
        content_data = cast("list[Any] | str", response.content)
        content = _extract_text(content_data)
        raw_response = response.model_dump()
        result = LLMResponse(
            content=content,
            model=response.model,
            raw=raw_response,
            elapsed_seconds=elapsed_seconds,
            tool_calls=normalize_tool_calls(raw_response),
        )
        self.log_io(direction="output", model=response.model, content=content)
        self.log_end("anthropic_generate", elapsed_seconds=elapsed_seconds)
        return result

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        thinking: Mapping[str, object] | None = None,
    ) -> LLMResponse:
        """Generate a completion using chat messages."""
        resolved_model = model or self._default_model
        self.log_start("anthropic_chat", model=resolved_model, message_count=len(messages))
        anthropic_messages = cast(
            "list[MessageParam]",
            [{"role": msg.role, "content": msg.content} for msg in messages],
        )
        request_kwargs = _build_messages_kwargs(
            resolved_model,
            anthropic_messages,
            temperature=self._temperature,
            thinking=thinking,
        )
        messages_client = cast("Any", self._client.messages)
        start = perf_counter()
        response = messages_client.create(**request_kwargs)
        elapsed_seconds = perf_counter() - start
        content_data = cast("list[Any] | str", response.content)
        content = _extract_text(content_data)
        raw_response = response.model_dump()
        result = LLMResponse(
            content=content,
            model=response.model,
            raw=raw_response,
            elapsed_seconds=elapsed_seconds,
            tool_calls=normalize_tool_calls(raw_response),
        )
        self.log_io(direction="output", model=response.model, content=content)
        self.log_end("anthropic_chat", elapsed_seconds=elapsed_seconds)
        return result

    def vision(
        self,
        prompt: str,
        image: ImageInput,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a completion using an image payload."""
        resolved_model = model or self._default_model
        self.log_start(
            "anthropic_vision",
            model=resolved_model,
            image_media_type=image.media_type,
        )
        self.log_io(direction="input", prompt=prompt, image_bytes=len(image.data))
        messages = cast(
            "list[MessageParam]",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image.media_type,
                                "data": image.data,
                            },
                        },
                    ],
                },
            ],
        )

        request_kwargs: dict[str, object] = {
            "model": resolved_model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": self._temperature,
        }
        if system_prompt is not None:
            request_kwargs["system"] = system_prompt

        messages_client = cast("Any", self._client.messages)
        start = perf_counter()
        response = messages_client.create(**request_kwargs)
        elapsed_seconds = perf_counter() - start
        content_data = cast("list[Any] | str", response.content)
        content = _extract_text(content_data)
        raw_response = response.model_dump()
        result = LLMResponse(
            content=content,
            model=response.model,
            raw=raw_response,
            elapsed_seconds=elapsed_seconds,
            tool_calls=normalize_tool_calls(raw_response),
        )
        self.log_io(direction="output", model=response.model, content=content)
        self.log_end("anthropic_vision", elapsed_seconds=elapsed_seconds)
        return result


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


def _build_messages_kwargs(
    model: str,
    messages: list[MessageParam],
    *,
    temperature: float,
    thinking: Mapping[str, object] | None = None,
) -> dict[str, object]:
    request_kwargs: dict[str, object] = {
        "model": model,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": temperature,
    }
    if thinking is not None:
        request_kwargs["thinking"] = dict(thinking)
    return request_kwargs
