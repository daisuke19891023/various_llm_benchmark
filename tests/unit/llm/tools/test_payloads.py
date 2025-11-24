"""Tests for provider tool payload helpers."""

from __future__ import annotations

from typing import Any, cast

from google.genai import types as genai_types

from various_llm_benchmark.llm.tools.payloads import (
    to_agents_sdk_tools_payload,
    to_agno_tools_payload,
    to_anthropic_tools_payload,
    to_gemini_tools_payload,
    to_google_adk_tools_payload,
    to_openai_tools_payload,
)
from various_llm_benchmark.llm.tools.registry import (
    CODE_EXECUTION_TAG,
    RETRIEVER_INPUT_SCHEMA,
    RETRIEVER_TAG,
    SHELL_TAG,
    WEB_SEARCH_TAG,
    NativeToolType,
    ToolCategory,
    ToolRegistration,
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


def _code_execution_tool() -> ToolRegistration:
    return ToolRegistration(
        id="code/execute",
        name="code-execute",
        description="Native code execution",
        input_schema={"type": "object"},
        tags={CODE_EXECUTION_TAG},
        native_type=NativeToolType.CODE_EXECUTION,
        handler=_noop,
        category=ToolCategory.BUILTIN,
    )


def _shell_tool() -> ToolRegistration:
    return ToolRegistration(
        id="shell/run",
        name="bash-execute",
        description="Execute shell commands",
        input_schema={"type": "object"},
        tags={SHELL_TAG},
        native_type=NativeToolType.SHELL,
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

    first_tool = payload[0]
    assert first_tool.function_declarations
    declarations = first_tool.function_declarations
    first_declaration = declarations[0]

    assert isinstance(first_tool, genai_types.Tool)
    assert isinstance(first_declaration, genai_types.FunctionDeclaration)
    assert first_declaration.name == "math-add"

    search_tool = payload[1]
    assert search_tool.google_search_retrieval is not None
    assert isinstance(search_tool.google_search_retrieval, genai_types.GoogleSearchRetrieval)


def test_agent_framework_helpers_delegate_to_llm_payloads() -> None:
    """Agent helpers should reuse LLM payload conversions."""
    tools = [_function_tool(), _web_search_tool()]

    assert to_agno_tools_payload(tools) == to_openai_tools_payload(tools)
    assert to_agents_sdk_tools_payload(tools) == to_openai_tools_payload(tools)
    google_adk_tools = to_google_adk_tools_payload(tools)
    gemini_tools = to_gemini_tools_payload(tools)

    assert len(google_adk_tools) == len(gemini_tools)
    assert isinstance(google_adk_tools[0], genai_types.Tool)
    assert google_adk_tools[0].function_declarations is not None
    assert gemini_tools[0].function_declarations is not None
    assert google_adk_tools[0].function_declarations[0].name == "math-add"
    assert google_adk_tools[0].function_declarations[0].name == gemini_tools[0].function_declarations[0].name
    assert google_adk_tools[1].google_search_retrieval is not None
    assert isinstance(google_adk_tools[1].google_search_retrieval, genai_types.GoogleSearchRetrieval)


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
    gemini_tool = gemini_payload[0]
    assert gemini_tool.function_declarations
    gemini_declarations = gemini_tool.function_declarations
    gemini_def = gemini_declarations[0]

    assert function_def["name"] == "openai-retriever"
    assert cast("dict[str, Any]", anthropic_def["input_schema"]) == RETRIEVER_INPUT_SCHEMA
    assert gemini_def.parameters_json_schema == RETRIEVER_INPUT_SCHEMA


def test_native_code_execution_payloads() -> None:
    """コード実行がネイティブツールとして扱われる."""
    code_execution = _code_execution_tool()

    openai_payload = to_openai_tools_payload([code_execution])
    anthropic_payload = to_anthropic_tools_payload([code_execution])
    gemini_payload = to_gemini_tools_payload([code_execution])

    assert openai_payload == [{"type": "code_interpreter"}]
    assert anthropic_payload == [
        {"type": "code_execution_20250825", "name": "code_execution"},
    ]
    assert len(gemini_payload) == 1
    gemini_tool = gemini_payload[0]
    assert gemini_tool.code_execution is not None
    assert isinstance(gemini_tool.code_execution, genai_types.ToolCodeExecution)


def test_shell_tools_convert_to_bash_payloads() -> None:
    """Shellツールは各プロバイダーのbashツールに変換される."""
    shell_tool = _shell_tool()

    openai_payload = to_openai_tools_payload([shell_tool])
    anthropic_payload = to_anthropic_tools_payload([shell_tool])

    assert openai_payload == [{"type": "bash"}]
    assert anthropic_payload == [{"type": "bash", "name": "bash-execute"}]


def test_shell_tool_provider_override_is_honored() -> None:
    """プロバイダーごとの上書きがShellツールにも適用される."""
    shell_tool = _shell_tool()
    shell_tool.provider_overrides = {
        "openai": {"type": "bash", "name": "custom-shell"},
        "anthropic": {"type": "bash", "name": "custom-shell"},
    }

    assert to_openai_tools_payload([shell_tool]) == [
        {"type": "bash", "name": "custom-shell"},
    ]
    assert to_anthropic_tools_payload([shell_tool]) == [
        {"type": "bash", "name": "custom-shell"},
    ]
