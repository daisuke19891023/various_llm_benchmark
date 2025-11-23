"""Tests for provider tool payload helpers."""

from __future__ import annotations

from typing import Any, cast

from various_llm_benchmark.llm.tools.payloads import (
    to_agents_sdk_tools_payload,
    to_agno_tools_payload,
    to_anthropic_tools_payload,
    to_gemini_tools_payload,
    to_google_adk_tools_payload,
    to_openai_tools_payload,
)
from various_llm_benchmark.llm.tools.registry import (
    NativeToolType,
    ToolCategory,
    ToolRegistration,
    RETRIEVER_INPUT_SCHEMA,
    RETRIEVER_TAG,
    WEB_SEARCH_TAG,
)


def _add(a: int, b: int) -> int:
    return a + b


def _noop() -> None:
    return None


def _function_tool() -> ToolRegistration:
    return ToolRegistration(
        id="math/add",
        name="math-add",
        description="Add two numbers",
        input_schema={"type": "object", "properties": {"a": {}, "b": {}}},
        tags={"math"},
        handler=_add,
        category=ToolCategory.EXTERNAL,
    )


def _web_search_tool() -> ToolRegistration:
    return ToolRegistration(
        id="web-search/openai",
        name="openai-web-search",
        description="Web search via OpenAI",
        input_schema={"type": "object"},
        tags={WEB_SEARCH_TAG, "provider:openai"},
        native_type=NativeToolType.WEB_SEARCH,
        handler=_noop,
        category=ToolCategory.BUILTIN,
    )


def _retriever_tool() -> ToolRegistration:
    return ToolRegistration(
        id="retriever/openai",
        name="openai-retriever",
        description="Hybrid retriever",
        input_schema=RETRIEVER_INPUT_SCHEMA,
        tags={RETRIEVER_TAG, "provider:openai"},
        native_type=NativeToolType.RETRIEVER,
        handler=_noop,
        category=ToolCategory.BUILTIN,
    )


def test_openai_payload_supports_functions_and_web_search() -> None:
    """OpenAI payload should convert functions and web search appropriately."""
    payload = to_openai_tools_payload([_function_tool(), _web_search_tool()])

    function_payload = cast("dict[str, Any]", payload[0]["function"])

    assert payload[0]["type"] == "function"
    assert function_payload["name"] == "math-add"
    assert payload[1] == {"type": "web_search"}


def test_anthropic_payload_converts_tool_fields() -> None:
    """Anthropic payload should map schema fields and web search."""
    payload = to_anthropic_tools_payload([_function_tool(), _web_search_tool()])

    first_tool = cast("dict[str, Any]", payload[0])
    input_schema = cast("dict[str, Any]", first_tool["input_schema"])

    assert first_tool["name"] == "math-add"
    assert input_schema["type"] == "object"
    assert payload[1] == {"type": "web_search"}


def test_gemini_payload_builds_function_declarations() -> None:
    """Gemini payload should use function declarations and search tool config."""
    payload = to_gemini_tools_payload([_function_tool(), _web_search_tool()])

    first_tool = cast("dict[str, Any]", payload[0])
    declarations = cast("list[object]", first_tool["function_declarations"])
    first_declaration = cast("dict[str, Any]", declarations[0])

    assert first_declaration["name"] == "math-add"
    assert payload[1] == {"google_search_retrieval": {}}


def test_agent_framework_helpers_delegate_to_llm_payloads() -> None:
    """Agent helpers should reuse LLM payload conversions."""
    tools = [_function_tool(), _web_search_tool()]

    assert to_agno_tools_payload(tools) == to_openai_tools_payload(tools)
    assert to_agents_sdk_tools_payload(tools) == to_openai_tools_payload(tools)
    assert to_google_adk_tools_payload(tools) == to_gemini_tools_payload(tools)


def test_web_search_detection_prefers_native_flag() -> None:
    """Native tool flag should mark web search even without the tag present."""
    web_search = _web_search_tool()
    web_search.tags = {"provider:openai"}

    assert to_openai_tools_payload([web_search]) == [{"type": "web_search"}]


def test_function_payload_includes_retriever_schema() -> None:
    """Retrieverツールのスキーマが各プロバイダー向けpayloadに反映される."""
    retriever = _retriever_tool()

    openai_payload = to_openai_tools_payload([retriever])
    anthropic_payload = to_anthropic_tools_payload([retriever])
    gemini_payload = to_gemini_tools_payload([retriever])

    function_def = cast("dict[str, Any]", cast("dict[str, Any]", openai_payload[0])["function"])
    anthropic_def = cast("dict[str, Any]", anthropic_payload[0])
    gemini_tool = cast("dict[str, Any]", gemini_payload[0])
    gemini_declarations = cast("list[object]", gemini_tool["function_declarations"])
    gemini_def = cast("dict[str, Any]", gemini_declarations[0])

    assert function_def["name"] == "openai-retriever"
    assert cast("dict[str, Any]", anthropic_def["input_schema"]) == RETRIEVER_INPUT_SCHEMA
    assert cast("dict[str, Any]", gemini_def["parameters"]) == RETRIEVER_INPUT_SCHEMA

