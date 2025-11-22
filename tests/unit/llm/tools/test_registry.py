"""Unit tests for the tool registry."""

from __future__ import annotations

import pytest

from various_llm_benchmark.llm.tools.registry import (
    ToolCategory,
    ToolRegistration,
    get_tool,
    list_tools,
    register_tool,
)


def _noop_handler() -> None:
    """Return nothing for registry tests."""


def test_register_and_retrieve_tool() -> None:
    """Registration should persist and be retrievable by id."""
    registration = ToolRegistration(
        id="test/tool",
        description="example",
        input_schema={"type": "object"},
        handler=_noop_handler,
        category=ToolCategory.BUILTIN,
    )
    register_tool(registration)

    retrieved = get_tool("test/tool")

    assert retrieved.id == "test/tool"
    assert retrieved.category is ToolCategory.BUILTIN


def test_list_tools_filters_by_category() -> None:
    """Tools can be filtered by category when listed."""
    external_tool = ToolRegistration(
        id="external/tool",
        description="external",
        input_schema={"type": "object"},
        handler=_noop_handler,
        category=ToolCategory.EXTERNAL,
    )
    register_tool(external_tool)

    builtin_tools = list_tools(category=ToolCategory.BUILTIN)
    assert all(tool.category is ToolCategory.BUILTIN for tool in builtin_tools)

    external_tools = list_tools(category=ToolCategory.EXTERNAL)
    assert {tool.id for tool in external_tools} == {"external/tool"}


def test_registering_duplicate_tool_is_rejected() -> None:
    """Duplicate tool identifiers raise a ValueError."""
    duplicate = ToolRegistration(
        id="duplicate/tool",
        description="first",
        input_schema={"type": "object"},
        handler=_noop_handler,
        category=ToolCategory.MCP,
    )
    register_tool(duplicate)

    with pytest.raises(ValueError, match="duplicate/tool"):
        register_tool(duplicate)


def test_category_mismatch_raises_lookup_error() -> None:
    """Retrieving with mismatched category should raise LookupError."""
    mismatched = ToolRegistration(
        id="mismatch/tool",
        description="mismatch",
        input_schema={"type": "object"},
        handler=_noop_handler,
        category=ToolCategory.BUILTIN,
    )
    register_tool(mismatched)

    with pytest.raises(LookupError):
        get_tool("mismatch/tool", category=ToolCategory.EXTERNAL)
