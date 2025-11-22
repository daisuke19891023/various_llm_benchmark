from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from various_llm_benchmark.llm.protocol import LLMClient
from various_llm_benchmark.models import ChatMessage, LLMResponse

if TYPE_CHECKING:
    from google.genai import Client


class GeminiLLMClient(LLMClient):
    """Adapter for Google Gemini API."""

    def __init__(self, client: Client, default_model: str, *, temperature: float = 0.7) -> None:
        """Create a Gemini client wrapper with defaults."""
        self._client = client
        self._default_model = default_model
        self._temperature = temperature

    def generate(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        """Generate a completion without history."""
        models_client = cast("Any", self._client.models)
        response: Any = models_client.generate_content(
            model=model or self._default_model,
            contents=prompt,
            config={"temperature": self._temperature},
        )
        content = _extract_text(response)
        model_name = _extract_model(response, model or self._default_model)
        return LLMResponse(content=content, model=model_name, raw=_dump_raw(response))

    def chat(
        self, messages: list[ChatMessage], *, model: str | None = None, system_instruction: str | None = None,
    ) -> LLMResponse:
        """Generate a completion using chat messages."""
        system_prompt, chat_messages = _extract_system_instruction(messages, system_instruction)
        gemini_messages: list[dict[str, object]] = [
            _to_gemini_message(message) for message in chat_messages
        ]
        models_client = cast("Any", self._client.models)
        request_kwargs: dict[str, object] = {
            "model": model or self._default_model,
            "contents": gemini_messages,
            "config": {"temperature": self._temperature},
        }
        if system_prompt:
            request_kwargs["system_instruction"] = system_prompt

        response: Any = models_client.generate_content(**request_kwargs)
        content = _extract_text(response)
        model_name = _extract_model(response, model or self._default_model)
        return LLMResponse(content=content, model=model_name, raw=_dump_raw(response))


def _to_gemini_message(message: ChatMessage) -> dict[str, object]:
    role = "model" if message.role == "assistant" else message.role
    return {"role": role, "parts": [message.content]}


def _extract_system_instruction(
    messages: list[ChatMessage], override: str | None,
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
