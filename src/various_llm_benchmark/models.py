"""Shared Pydantic data models used across LLM and agent providers."""

from __future__ import annotations

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """Represents a single chat message."""

    role: str
    content: str


class LLMResponse(BaseModel):
    """Normalized response from a provider."""

    content: str
    model: str
    raw: dict[str, object]
