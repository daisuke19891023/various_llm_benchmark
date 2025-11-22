from __future__ import annotations

import typer

from various_llm_benchmark.models import ChatMessage

MISSING_HISTORY_FORMAT_ERROR = "履歴は 'role:content' 形式で指定してください。"


def parse_history(history: list[str]) -> list[ChatMessage]:
    """Convert history strings into structured chat messages."""
    messages: list[ChatMessage] = []
    for entry in history:
        if ":" not in entry:
            raise typer.BadParameter(MISSING_HISTORY_FORMAT_ERROR)
        role, content = entry.split(":", 1)
        messages.append(ChatMessage(role=role.strip(), content=content.strip()))
    return messages


def build_messages(prompt: str, history: list[str], *, system_prompt: str | None = None) -> list[ChatMessage]:
    """Append the current prompt to parsed history messages with an optional system prompt."""
    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage(role="system", content=system_prompt))

    messages.extend(parse_history(history))
    messages.append(ChatMessage(role="user", content=prompt))
    return messages
