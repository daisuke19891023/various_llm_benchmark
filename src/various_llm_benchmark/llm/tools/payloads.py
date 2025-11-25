"""Helpers for converting tool registrations into provider payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from google.genai import types as genai_types

from various_llm_benchmark.llm.tools.registry import (
    CODE_EXECUTION_TAG,
    SHELL_TAG,
    WEB_SEARCH_TAG,
    NativeToolType,
    ToolRegistration,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _is_web_search(tool: ToolRegistration) -> bool:
    return tool.native_type is NativeToolType.WEB_SEARCH or WEB_SEARCH_TAG in tool.tags


def _is_code_execution(tool: ToolRegistration) -> bool:
    return tool.native_type is NativeToolType.CODE_EXECUTION or CODE_EXECUTION_TAG in tool.tags


def _is_shell(tool: ToolRegistration) -> bool:
    return tool.native_type is NativeToolType.SHELL or SHELL_TAG in tool.tags


def _function_payload(tool: ToolRegistration) -> dict[str, object]:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
    }


def to_openai_tools_payload(tools: Sequence[ToolRegistration]) -> list[dict[str, object]]:
    """Convert tool registrations to OpenAI Responses `tools` payload."""
    payload: list[dict[str, object]] = []
    for tool in tools:
        override = tool.provider_overrides.get("openai")
        if isinstance(override, dict):
            payload.append(cast("dict[str, object]", override))
            continue
        if _is_web_search(tool):
            payload.append({"type": "web_search"})
            continue
        if _is_shell(tool):
            payload.append({"type": "bash"})
            continue
        if _is_code_execution(tool):
            payload.append({"type": "code_interpreter"})
            continue
        payload.append(
            {
                "type": "function",
                "function": _function_payload(tool),
            },
        )
    return payload


def to_anthropic_tools_payload(tools: Sequence[ToolRegistration]) -> list[dict[str, object]]:
    """Convert tool registrations to Claude Messages API payload."""
    payload: list[dict[str, object]] = []
    for tool in tools:
        override = tool.provider_overrides.get("anthropic")
        if isinstance(override, dict):
            payload.append(cast("dict[str, object]", override))
            continue
        if _is_web_search(tool):
            payload.append({"type": "web_search"})
            continue
        if _is_shell(tool):
            payload.append({"type": "bash", "name": tool.name})
            continue
        if _is_code_execution(tool):
            payload.append({"type": "code_execution_20250825", "name": "code_execution"})
            continue
        payload.append(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            },
        )
    return payload


def to_gemini_tools_payload(tools: Sequence[ToolRegistration]) -> list[genai_types.Tool]:
    """Convert tool registrations to Gemini `tools` payload."""
    payload: list[genai_types.Tool] = []
    for tool in tools:
        override = tool.provider_overrides.get("gemini")
        if isinstance(override, genai_types.Tool):
            payload.append(override)
            continue
        if isinstance(override, dict):
            payload.append(genai_types.Tool.model_validate(override))
            continue
        if _is_web_search(tool):
            payload.append(
                genai_types.Tool(
                    google_search_retrieval=genai_types.GoogleSearchRetrieval(),
                ),
            )
            continue
        if _is_code_execution(tool):
            payload.append(
                genai_types.Tool(
                    code_execution=genai_types.ToolCodeExecution(),
                ),
            )
            continue
        function_declaration = genai_types.FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters_json_schema=tool.input_schema,
        )
        payload.append(genai_types.Tool(function_declarations=[function_declaration]))
    return payload


def to_agno_tools_payload(tools: Sequence[ToolRegistration]) -> list[dict[str, object]]:
    """Return tool payloads suitable for Agno agents."""
    return to_openai_tools_payload(tools)


def to_agents_sdk_tools_payload(tools: Sequence[ToolRegistration]) -> list[dict[str, object]]:
    """Return tool payloads suitable for the OpenAI Agents SDK."""
    return to_openai_tools_payload(tools)


def to_google_adk_tools_payload(tools: Sequence[ToolRegistration]) -> list[genai_types.Tool]:
    """Return tool payloads suitable for Google ADK agents."""
    return to_gemini_tools_payload(tools)
