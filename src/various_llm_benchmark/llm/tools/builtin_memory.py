"""In-memory message store exposed as tool functions."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from various_llm_benchmark.llm.tools.registry import ToolCategory, ToolRegistration, register_tool


@dataclass(frozen=True)
class MemoryEntry:
    """Represents a stored chat message."""

    role: str
    content: str


class InMemoryStore:
    """Thread-safe in-memory message store."""

    def __init__(self) -> None:
        """Initialize empty storage containers."""
        self._messages: list[MemoryEntry] = []
        self._lock = Lock()

    def append(self, role: str, content: str) -> MemoryEntry:
        """Append a message to the store."""
        entry = MemoryEntry(role=role, content=content)
        with self._lock:
            self._messages.append(entry)
        return entry

    def search(self, query: str, *, limit: int = 5) -> list[MemoryEntry]:
        """Return messages containing the query string (case-insensitive)."""
        normalized_query = query.casefold()
        with self._lock:
            matches = [
                message
                for message in reversed(self._messages)
                if normalized_query in message.content.casefold()
            ]
        return list(reversed(matches[:limit]))

    def snapshot(self) -> list[MemoryEntry]:
        """Return a copy of all messages in insertion order."""
        with self._lock:
            return list(self._messages)

    def clear(self) -> None:
        """Remove all messages from the store."""
        with self._lock:
            self._messages.clear()


_store = InMemoryStore()


def append_memory(role: str, content: str) -> dict[str, str]:
    """Append a message to the shared store and return it."""
    entry = _store.append(role, content)
    return {"role": entry.role, "content": entry.content}


def search_memory(query: str, limit: int = 5) -> dict[str, list[dict[str, str]]]:
    """Search the shared store for messages matching the query."""
    matches = _store.search(query, limit=limit)
    return {"matches": [{"role": match.role, "content": match.content} for match in matches]}


def get_store() -> InMemoryStore:
    """Return the singleton in-memory store for testing or inspection."""
    return _store


def reset_store() -> None:
    """Clear stored messages (primarily for tests)."""
    _store.clear()


def _register_tools() -> None:
    """Register built-in memory tools with the registry."""
    append_schema = {
        "type": "object",
        "properties": {
            "role": {
                "type": "string",
                "description": "メッセージの役割 (user, assistant, system など)",
            },
            "content": {"type": "string", "description": "保存するテキスト"},
        },
        "required": ["role", "content"],
    }
    search_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "検索キーワード"},
            "limit": {"type": "integer", "description": "返却件数の上限", "minimum": 1},
        },
        "required": ["query"],
    }

    for registration in (
        ToolRegistration(
            id="memory/append",
            name="memory-append",
            description="インメモリストアへメッセージを追記する",
            input_schema=append_schema,
            tags={"memory", "builtin"},
            handler=append_memory,
            category=ToolCategory.BUILTIN,
        ),
        ToolRegistration(
            id="memory/search",
            name="memory-search",
            description="インメモリストアからメッセージを検索する",
            input_schema=search_schema,
            tags={"memory", "builtin"},
            handler=search_memory,
            category=ToolCategory.BUILTIN,
        ),
    ):
        try:
            register_tool(registration)
        except ValueError:
            continue


_register_tools()

