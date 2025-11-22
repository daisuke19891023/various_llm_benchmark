"""Selection utilities for filtering registered tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from various_llm_benchmark.llm.tools.registry import ToolCategory, ToolRegistration, list_tools

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class ToolSelector:
    """Select tools from the registry based on metadata."""

    def __init__(self, *, tools: Iterable[ToolRegistration] | None = None) -> None:
        """Initialize the selector with a snapshot of tools."""
        self._tools = list(tools) if tools is not None else list_tools()

    def select(
        self,
        *,
        category: ToolCategory | None = None,
        names: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        ids: Sequence[str] | None = None,
    ) -> list[ToolRegistration]:
        """Return tools matching all provided filters."""
        selected: list[ToolRegistration] = []
        name_filter = set(names or [])
        tag_filter = set(tags or [])
        id_filter = set(ids or [])

        for tool in self._tools:
            if category is not None and tool.category is not category:
                continue
            if name_filter and tool.name not in name_filter:
                continue
            if id_filter and tool.id not in id_filter:
                continue
            if tag_filter and not tag_filter.issubset(tool.tags):
                continue
            selected.append(tool)
        return selected

    def select_one(
        self,
        *,
        category: ToolCategory | None = None,
        names: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        ids: Sequence[str] | None = None,
    ) -> ToolRegistration:
        """Return exactly one tool or raise a ValueError when unmatched."""
        matches = self.select(category=category, names=names, tags=tags, ids=ids)
        if not matches:
            msg = "No tools matched the given filters."
            raise ValueError(msg)
        if len(matches) > 1:
            msg = "Multiple tools matched the given filters."
            raise ValueError(msg)
        return matches[0]

