"""Shared Pydantic data models used across LLM and agent providers."""

from __future__ import annotations

import json
from typing import Any, TypedDict, cast
from collections.abc import Iterator, Mapping

from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    """Represents a single chat message."""

    role: str
    content: str


class MediaInput(BaseModel):
    """Represents a binary media payload encoded for LLM consumption."""

    media_type: str
    data: str

    def as_data_url(self) -> str:
        """Return the media as a data URL for providers that expect URLs."""
        return f"data:{self.media_type};base64,{self.data}"


class ImageInput(MediaInput):
    """Represents an image payload encoded for LLM consumption."""


class ToolCall(BaseModel):
    """Normalized tool invocation data extracted from provider responses."""

    name: str
    arguments: object | None = None
    output: str | None = None
    call_id: str | None = None
    call_type: str | None = None


class ToolCallPayload(TypedDict, total=False):
    """Raw tool call payload compatible across providers."""

    function: Mapping[str, object | None]
    name: object | None
    arguments: object | None
    output: object | None
    id: object | None
    type: object | None


def _empty_tool_calls() -> list[ToolCall]:
    return []


class LLMResponse(BaseModel):
    """Normalized response from a provider."""

    content: str
    model: str
    raw: dict[str, object]
    elapsed_seconds: float | None = None
    call_count: int = 1
    tool_calls: list[ToolCall] = Field(default_factory=_empty_tool_calls)
    tools: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _populate_tools(self) -> LLMResponse:
        """Ensure tools list mirrors tool call names when absent."""
        if not self.tools and self.tool_calls:
            self.tools = sorted({call.name for call in self.tool_calls if call.name})
        return self


def normalize_tool_calls(raw: Mapping[str, object]) -> list[ToolCall]:
    """Extract tool call details from provider-specific response payloads."""
    normalized_calls: list[ToolCall] = []
    for candidate in _iter_tool_call_candidates(raw):
        parsed_call = _parse_tool_call(candidate)
        if parsed_call is not None:
            normalized_calls.append(parsed_call)
    return normalized_calls


def _iter_tool_call_candidates(value: object) -> Iterator[Mapping[str, object | None]]:
    stack: list[object] = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            mapping_value: dict[str, object] = dict(cast("Mapping[str, object]", current))
            tool_calls_value = mapping_value.get("tool_calls")
            if isinstance(tool_calls_value, list):
                for call in cast("list[object]", tool_calls_value):
                    if isinstance(call, Mapping):
                        yield cast("Mapping[str, object | None]", call)
            stack.extend(list(mapping_value.values()))
        elif isinstance(current, list):
            stack.extend(list(cast("list[object]", current)))


def _parse_tool_call(candidate: Mapping[str, object | None]) -> ToolCall | None:
    payload: ToolCallPayload = cast("ToolCallPayload", candidate)
    function_payload = payload.get("function")
    if isinstance(function_payload, Mapping):
        function_name = str(function_payload.get("name"))
        output_value = payload.get("output")
        call_id_value = payload.get("id")
        call_type_value = payload.get("type")
        return ToolCall(
            name=function_name,
            arguments=_safe_arguments(function_payload.get("arguments")),
            output=_stringify(output_value),
            call_id=_stringify(call_id_value),
            call_type=_stringify(call_type_value),
        )

    name_value = payload.get("name")
    if name_value is None:
        return None

    output_value = payload.get("output")
    call_id_value = payload.get("id")
    call_type_value = payload.get("type")
    arguments_value = payload.get("arguments")
    return ToolCall(
        name=str(name_value),
        arguments=_safe_arguments(arguments_value),
        output=_stringify(output_value),
        call_id=_stringify(call_id_value),
        call_type=_stringify(call_type_value),
    )


def _safe_arguments(arguments: object | None) -> object | None:
    if not isinstance(arguments, str):
        return arguments
    try:
        return json.loads(arguments)
    except Exception:
        return arguments


def _stringify(value: Any | None) -> str | None:
    return None if value is None else str(value)
