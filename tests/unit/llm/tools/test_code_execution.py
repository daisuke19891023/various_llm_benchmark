"""Tests for safe code execution tool registration and payloads."""

from __future__ import annotations


import pytest

from google.genai import types as genai_types

from various_llm_benchmark.llm.tools.builtin_memory import reset_store
from various_llm_benchmark.llm.tools.code_execution import (
    ALLOWED_LANGUAGES,
    SafeCodeExecutor,
)
from various_llm_benchmark.llm.tools.payloads import (
    to_anthropic_tools_payload,
    to_gemini_tools_payload,
    to_openai_tools_payload,
)
from various_llm_benchmark.llm.tools.selector import ToolSelector


def test_executor_runs_python_and_captures_output() -> None:
    """Python snippets should be executed and outputs returned."""
    executor = SafeCodeExecutor(timeout_seconds=2)
    result = executor.run("python", "print('hello')")

    assert result.exit_code == 0
    assert result.stdout.strip() == "hello"


def test_executor_rejects_unknown_language() -> None:
    """Unsupported languages should raise ValueError."""
    executor = SafeCodeExecutor()

    with pytest.raises(ValueError, match="Unsupported language"):
        executor.run("javascript", "console.log('hi')")


def test_executor_can_chain_registered_tools_with_helpers() -> None:
    """Code execution can orchestrate registered tools and post-process outputs."""
    reset_store()
    executor = SafeCodeExecutor()
    code = """
call_tool("memory/append", role="user", content="first")
call_tool("memory/append", role="assistant", content="second")
results = call_tool("memory/search", query="", limit=5)
matches = coerce_list(results.get("matches"))
limited = limit_items(matches, 1)
print(len(matches))
print(limited[0]["content"])
"""

    result = executor.run(
        "python",
        code,
        allowed_tool_ids=["memory/append", "memory/search"],
    )

    assert result.exit_code == 0
    output_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert output_lines == ["2", "first"]


def test_executor_blocks_unlisted_tools() -> None:
    """Attempting to call a non-whitelisted tool should surface an error."""
    executor = SafeCodeExecutor()
    code = """
call_tool("memory/append", role="user", content="not allowed")
"""

    result = executor.run("python", code, allowed_tool_ids=["memory/search"])

    assert result.exit_code == 1
    assert "not allowed" not in result.stdout
    assert "not allowed" in result.stderr


def test_payload_overrides_are_used_for_providers() -> None:
    """Registered tool should surface provider-specific payload overrides."""
    selector = ToolSelector()
    code_tool = selector.select_one(ids=["code/execute"], category=None)

    assert set(ALLOWED_LANGUAGES) == {"python"}

    openai_payload = to_openai_tools_payload([code_tool])
    anthropic_payload = to_anthropic_tools_payload([code_tool])
    gemini_payload = to_gemini_tools_payload([code_tool])

    assert openai_payload == [{"type": "code_interpreter"}]
    assert anthropic_payload == [
        {"type": "code_execution_20250825", "name": "code_execution"},
    ]
    assert len(gemini_payload) == 1
    gemini_tool = gemini_payload[0]
    assert isinstance(gemini_tool, genai_types.Tool)
    assert gemini_tool.code_execution is not None
    assert isinstance(gemini_tool.code_execution, genai_types.ToolCodeExecution)

