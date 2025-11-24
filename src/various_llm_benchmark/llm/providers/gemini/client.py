from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from various_llm_benchmark.llm.protocol import LLMClient
from various_llm_benchmark.logger import BaseComponent
from various_llm_benchmark.models import ChatMessage, ImageInput, LLMResponse, MediaInput, normalize_tool_calls

if TYPE_CHECKING:
    from google.genai import Client


class GeminiLLMClient(LLMClient, BaseComponent):
    """Adapter for Google Gemini API."""

    def __init__(
        self,
        client: Client,
        default_model: str,
        *,
        temperature: float = 0.7,
        thinking_level: str | None = None,
    ) -> None:
        """Create a Gemini client wrapper with defaults."""
        self._client = client
        self._default_model = default_model
        self._temperature = temperature
        self._thinking_level = thinking_level

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> LLMResponse:
        """Generate a completion without history."""
        resolved_model = model or self._default_model
        self.log_start("gemini_generate", model=resolved_model)
        self.log_io(direction="input", prompt=prompt)
        models_client = cast("Any", self._client.models)
        start = perf_counter()
        response: Any = models_client.generate_content(
            model=resolved_model,
            contents=prompt,
            config=_build_config(self._temperature, self._thinking_level, thinking_level),
        )
        elapsed_seconds = perf_counter() - start
        content = _extract_text(response)
        model_name = _extract_model(response, resolved_model)
        raw_response = _dump_raw(response)
        result = LLMResponse(
            content=content,
            model=model_name,
            raw=raw_response,
            elapsed_seconds=elapsed_seconds,
            tool_calls=normalize_tool_calls(raw_response),
        )
        self.log_io(direction="output", model=model_name, content=content)
        self.log_end("gemini_generate", elapsed_seconds=elapsed_seconds)
        return result

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        system_instruction: str | None = None,
        thinking_level: str | None = None,
    ) -> LLMResponse:
        """Generate a completion using chat messages."""
        resolved_model = model or self._default_model
        # Extract system prompt (ignored for Gemini) and remaining chat messages
        _, chat_messages = _extract_system_instruction(messages, system_instruction)
        # Gemini does not support system role in contents; we ignore it.
        self.log_start(
            "gemini_chat",
            model=resolved_model,
            message_count=len(chat_messages),
            has_system=False,
        )
        # Convert messages to a simple text format for Gemini
        # Gemini API expects contents as string or simple format, not structured messages
        contents_text = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_messages])
        models_client = cast("Any", self._client.models)
        request_kwargs: dict[str, object] = {
            "model": resolved_model,
            "contents": contents_text,
            "config": _build_config(self._temperature, self._thinking_level, thinking_level),
        }

        start = perf_counter()
        response: Any = models_client.generate_content(**request_kwargs)
        elapsed_seconds = perf_counter() - start
        content = _extract_text(response)
        model_name = _extract_model(response, resolved_model)
        raw_response = _dump_raw(response)
        result = LLMResponse(
            content=content,
            model=model_name,
            raw=raw_response,
            elapsed_seconds=elapsed_seconds,
            tool_calls=normalize_tool_calls(raw_response),
        )
        self.log_io(direction="output", model=model_name, content=content)
        self.log_end("gemini_chat", elapsed_seconds=elapsed_seconds)
        return result

    def vision(
        self,
        prompt: str,
        image: ImageInput,
        *,
        model: str | None = None,
        thinking_level: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response that includes image context."""
        resolved_model = model or self._default_model
        self.log_start(
            "gemini_vision",
            model=resolved_model,
            image_media_type=image.media_type,
        )
        self.log_io(direction="input", prompt=prompt, image_bytes=len(image.data))
        models_client = cast("Any", self._client.models)
        request_kwargs: dict[str, object] = {
            "model": resolved_model,
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": image.media_type,
                                "data": image.data,
                            },
                        },
                    ],
                },
            ],
            "config": _build_config(self._temperature, self._thinking_level, thinking_level),
        }
        # system_prompt is ignored (not supported in current API version)

        start = perf_counter()
        response: Any = models_client.generate_content(**request_kwargs)
        elapsed_seconds = perf_counter() - start
        content = _extract_text(response)
        model_name = _extract_model(response, resolved_model)
        raw_response = _dump_raw(response)
        result = LLMResponse(
            content=content,
            model=model_name,
            raw=raw_response,
            elapsed_seconds=elapsed_seconds,
            tool_calls=normalize_tool_calls(raw_response),
        )
        self.log_io(direction="output", model=model_name, content=content)
        self.log_end("gemini_vision", elapsed_seconds=elapsed_seconds)
        return result

    def multimodal(
        self,
        prompt: str,
        media: Sequence[MediaInput],
        *,
        model: str | None = None,
        thinking_level: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response that includes audio or video context."""
        if not media:
            message = "At least one media input is required for Gemini multimodal calls"
            raise ValueError(message)

        resolved_model = model or self._default_model
        self.log_start(
            "gemini_multimodal",
            model=resolved_model,
            media_count=len(media),
            media_types=[item.media_type for item in media],
        )
        self.log_io(direction="input", prompt=prompt, media_items=len(media))
        models_client = cast("Any", self._client.models)
        parts: list[dict[str, object]] = [{"text": prompt}]
        parts.extend({"inline_data": {"mime_type": item.media_type, "data": item.data}} for item in media)

        request_kwargs: dict[str, object] = {
            "model": resolved_model,
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                },
            ],
            "config": _build_config(self._temperature, self._thinking_level, thinking_level),
        }
        # system_prompt is ignored (not supported in current API version)

        start = perf_counter()
        response: Any = models_client.generate_content(**request_kwargs)
        elapsed_seconds = perf_counter() - start
        content = _extract_text(response)
        model_name = _extract_model(response, resolved_model)
        raw_response = _dump_raw(response)
        result = LLMResponse(
            content=content,
            model=model_name,
            raw=raw_response,
            elapsed_seconds=elapsed_seconds,
            tool_calls=normalize_tool_calls(raw_response),
        )
        self.log_io(direction="output", model=model_name, content=content)
        self.log_end("gemini_multimodal", elapsed_seconds=elapsed_seconds)
        return result


def _build_config(
    temperature: float,
    default_thinking_level: str | None,
    override_thinking_level: str | None,
) -> dict[str, object]:
    config: dict[str, object] = {"temperature": temperature}
    thinking_level = override_thinking_level or default_thinking_level
    if thinking_level:
        # NOTE: thinking_level is not yet supported in google.genai 1.52.0
        # When supported, add: config["thinking_level"] = thinking_level
        pass
    return config


def _extract_system_instruction(
    messages: list[ChatMessage],
    override: str | None,
) -> tuple[str | None, list[ChatMessage]]:
    system_messages = [msg for msg in messages if msg.role == "system"]
    non_system = [msg for msg in messages if msg.role != "system"]
    if not override and not system_messages:
        return None, non_system

    parts: list[str] = []
    if override:
        parts.append(override)
    if system_messages:
        parts.extend(msg.content for msg in system_messages)

    combined = "\n".join(parts)
    return combined, non_system


def _extract_model(response: Any, fallback: str) -> str:
    model_name = getattr(response, "model", None)
    return model_name if isinstance(model_name, str) else fallback


def _dump_raw(response: Any) -> dict[str, object]:
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return cast("dict[str, object]", dumped) if isinstance(dumped, Mapping) else {"data": dumped}

    to_dict = getattr(response, "to_dict", None)
    if callable(to_dict):
        dumped = to_dict()
        return cast("dict[str, object]", dumped) if isinstance(dumped, Mapping) else {"data": dumped}

    if hasattr(response, "__dict__"):
        return cast("dict[str, object]", dict(response.__dict__))

    return {"raw": str(response)}


def _extract_text(response: Any) -> str:
    text_attr = getattr(response, "text", None)
    if isinstance(text_attr, str):
        return text_attr

    candidates = cast("list[Any] | None", getattr(response, "candidates", None))
    if isinstance(candidates, list) and candidates:
        for candidate in candidates:
            candidate_text = getattr(candidate, "text", None)
            if isinstance(candidate_text, str):
                return candidate_text
            content = getattr(candidate, "content", None)
            content_text = _extract_text(content) if content is not None else None
            if content_text:
                return content_text

    content_attr = getattr(response, "content", None)
    if isinstance(content_attr, Mapping):
        mapping_content = cast("Mapping[str, object]", content_attr)
        return _extract_from_mapping(mapping_content)
    if isinstance(content_attr, list):
        content_list = cast("list[Any]", content_attr)
        joined_content = [_extract_text(item) for item in content_list]
        return "".join(joined_content)

    return str(response)


def _extract_from_mapping(content: Mapping[str, object]) -> str:
    parts = content.get("parts")
    if isinstance(parts, list) and parts:
        parts_list = cast("list[object]", parts)
        text_parts: list[str] = []
        for part in parts_list:
            if isinstance(part, Mapping):
                mapping_part = cast("Mapping[str, object]", part)
                text = mapping_part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
                    continue
                text_parts.append(str(dict(mapping_part)))
                continue
            text_parts.append(str(part))
        if text_parts:
            return "".join(text_parts)

    text_value = content.get("text")
    if isinstance(text_value, str):
        return text_value

    return str(dict(content))


def extract_gemini_text(response: Any) -> str:
    """Public wrapper for parsing Gemini response content."""
    return _extract_text(response)
