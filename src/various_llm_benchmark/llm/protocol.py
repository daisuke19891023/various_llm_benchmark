from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """Represents a single chat message."""

    role: str
    content: str


class LLMResponse(BaseModel):
    """Normalized response from an LLM provider."""

    content: str
    model: str
    raw: dict[str, object]


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM providers that support text and chat generation."""

    def generate(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        """Generate a response for a single prompt."""
        ...

    def chat(self, messages: list[ChatMessage], *, model: str | None = None) -> LLMResponse:
        """Generate a response using a chat-style message history."""
        ...
