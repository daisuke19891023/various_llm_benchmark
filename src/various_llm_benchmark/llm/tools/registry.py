"""Tool registry for model-native and external integrations."""

from __future__ import annotations

from enum import StrEnum
import typing
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolCategory(StrEnum):
    """Supported tool source categories."""

    BUILTIN = "builtin"
    MCP = "mcp"
    EXTERNAL = "external"


class ToolRegistration(BaseModel):
    """Structured metadata for a tool entry in the registry."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(min_length=1)
    description: str
    input_schema: dict[str, Any]
    handler: typing.Callable[..., Any]
    category: ToolCategory


_registry: dict[str, ToolRegistration] = {}


def register_tool(registration: ToolRegistration) -> None:
    """Register a new tool, rejecting duplicate identifiers."""
    if registration.id in _registry:
        msg = f"Tool '{registration.id}' is already registered."
        raise ValueError(msg)
    if not callable(registration.handler):
        msg = f"Handler for tool '{registration.id}' must be callable."
        raise TypeError(msg)
    _registry[registration.id] = registration


def get_tool(tool_id: str, *, category: ToolCategory | None = None) -> ToolRegistration:
    """Return a registered tool, optionally scoped by category."""
    try:
        registration = _registry[tool_id]
    except KeyError as exc:
        msg = f"Tool '{tool_id}' is not registered."
        raise KeyError(msg) from exc

    if category is not None and registration.category != category:
        msg = (
            f"Tool '{tool_id}' is registered under category '{registration.category}', "
            f"not '{category}'."
        )
        raise LookupError(msg)

    return registration


def list_tools(*, category: ToolCategory | None = None) -> list[ToolRegistration]:
    """Return all registered tools, filtered by category when provided."""
    if category is None:
        return list(_registry.values())
    return [tool for tool in _registry.values() if tool.category == category]
