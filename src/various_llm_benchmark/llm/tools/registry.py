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


class NativeToolType(StrEnum):
    """Known LLM-native tools handled by the provider itself."""

    WEB_SEARCH = "web_search"
    RETRIEVER = "retriever"
    CODE_EXECUTION = "code_execution"
    SHELL = "shell"


class ToolRegistration(BaseModel):
    """Structured metadata for a tool entry in the registry."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str
    input_schema: dict[str, Any]
    tags: set[str] = Field(default_factory=set)
    native_type: NativeToolType | None = None
    provider_overrides: dict[str, object] = Field(default_factory=dict)
    handler: typing.Callable[..., Any]
    category: ToolCategory


WEB_SEARCH_TAG = "web_search"
CODE_EXECUTION_TAG = "code_execution"
SHELL_TAG = "shell"
RETRIEVER_TAG = "retriever"
RETRIEVER_TOOL_NAMESPACE = "retriever"
RETRIEVER_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "検索クエリ"},
        "model": {"type": "string", "description": "埋め込みモデルを上書き"},
        "top_k": {
            "type": "integer",
            "minimum": 1,
            "description": "検索する件数の上限",
        },
        "threshold": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "スコアの下限値",
        },
        "timeout": {
            "type": "number",
            "minimum": 0,
            "description": "DBアクセスや埋め込み取得のタイムアウト (秒)",
            "default": 5.0,
        },
    },
    "required": ["query"],
}


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
