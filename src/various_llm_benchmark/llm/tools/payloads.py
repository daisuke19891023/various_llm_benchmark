"""Helpers for converting tool registrations into provider payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from various_llm_benchmark.llm.tools.registry import (
    NativeToolType,
    ToolRegistration,
    WEB_SEARCH_TAG,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _is_web_search(tool: ToolRegistration) -> bool:
    return tool.native_type is NativeToolType.WEB_SEARCH or WEB_SEARCH_TAG in tool.tags


def to_openai_tools_payload(tools: Sequence[ToolRegistration]) -> list[dict[str, object]]:
    """Convert tool registrations to OpenAI Responses `tools` payload."""
    payload: list[dict[str, object]] = []
    for tool in tools:
        if _is_web_search(tool):
            payload.append({"type": "web_search"})
            continue
        payload.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            },
        )
    return payload


def to_anthropic_tools_payload(tools: Sequence[ToolRegistration]) -> list[dict[str, object]]:
    """Convert tool registrations to Claude Messages API payload."""
    payload: list[dict[str, object]] = []
    for tool in tools:
        if _is_web_search(tool):
            payload.append({"type": "web_search"})
            continue
        payload.append(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            },
        )
    return payload


def to_gemini_tools_payload(tools: Sequence[ToolRegistration]) -> list[dict[str, object]]:
    """Convert tool registrations to Gemini `tools` payload."""
    payload: list[dict[str, object]] = []
    for tool in tools:
        if _is_web_search(tool):
            payload.append({"google_search_retrieval": {}})
            continue
        payload.append(
            {
                "function_declarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                ],
            },
        )
    return payload


def to_agno_tools_payload(tools: Sequence[ToolRegistration]) -> list[dict[str, object]]:
    """Return tool payloads suitable for Agno agents."""
    return to_openai_tools_payload(tools)


def to_agents_sdk_tools_payload(tools: Sequence[ToolRegistration]) -> list[dict[str, object]]:
    """Return tool payloads suitable for the OpenAI Agents SDK."""
    return to_openai_tools_payload(tools)


def to_google_adk_tools_payload(tools: Sequence[ToolRegistration]) -> list[dict[str, Any]]:
    """Return tool payloads suitable for Google ADK agents."""
    return to_gemini_tools_payload(tools)

