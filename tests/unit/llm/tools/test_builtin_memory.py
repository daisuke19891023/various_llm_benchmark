"""Tests for the built-in in-memory store tool registrations."""

from __future__ import annotations

from various_llm_benchmark.llm.tools.builtin_memory import append_memory, reset_store, search_memory
from various_llm_benchmark.llm.tools.registry import ToolCategory
from various_llm_benchmark.llm.tools.selector import ToolSelector


def test_append_and_search_roundtrip() -> None:
    """Messages should be persisted and searchable via the tool handlers."""
    reset_store()

    append_memory("user", "first message")
    append_memory("assistant", "second reply")

    results = search_memory("reply", limit=5)

    assert results["matches"]
    assert results["matches"][0]["content"] == "second reply"


def test_selector_returns_memory_tools() -> None:
    """Tool selector should expose both memory append and search tools."""
    selector = ToolSelector()
    memory_tools = selector.select(category=ToolCategory.BUILTIN, tags=["memory"])
    names = {tool.name for tool in memory_tools}

    assert names.issuperset({"memory-append", "memory-search"})

