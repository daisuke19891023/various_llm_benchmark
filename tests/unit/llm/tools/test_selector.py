"""Tests for the ToolSelector helper."""

from __future__ import annotations

import pytest

from various_llm_benchmark.llm.tools.registry import ToolCategory, ToolRegistration
from various_llm_benchmark.llm.tools.selector import ToolSelector


def _tool(id_suffix: str, *, category: ToolCategory, tags: set[str]) -> ToolRegistration:
    return ToolRegistration(
        id=f"tool/{id_suffix}",
        name=f"tool-{id_suffix}",
        description="desc",
        input_schema={"type": "object"},
        tags=tags,
        handler=lambda: None,
        category=category,
    )


def test_select_filters_by_category_and_tags() -> None:
    """Selector should combine category and tag filters."""
    tools = [
        _tool("a", category=ToolCategory.BUILTIN, tags={"alpha", "shared"}),
        _tool("b", category=ToolCategory.MCP, tags={"beta", "shared"}),
        _tool("c", category=ToolCategory.BUILTIN, tags={"gamma"}),
    ]

    selector = ToolSelector(tools=tools)
    selected = selector.select(category=ToolCategory.BUILTIN, tags=["shared"])

    assert [tool.id for tool in selected] == ["tool/a"]


def test_select_one_requires_single_match() -> None:
    """Select one should raise when zero or multiple tools match."""
    tools = [
        _tool("a", category=ToolCategory.EXTERNAL, tags=set()),
        _tool("b", category=ToolCategory.EXTERNAL, tags={"x"}),
    ]

    selector = ToolSelector(tools=tools)
    with pytest.raises(ValueError, match="No tools matched"):
        selector.select_one(names=["missing"])

    with pytest.raises(ValueError, match="Multiple tools matched"):
        selector.select_one(category=ToolCategory.EXTERNAL)

    chosen = selector.select_one(category=ToolCategory.EXTERNAL, tags=["x"])
    assert chosen.id == "tool/b"

