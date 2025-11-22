"""Protocols for agent providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from various_llm_benchmark.models import ChatMessage, ImageInput, LLMResponse


@runtime_checkable
class AgentClient(Protocol):
    """Protocol for agent providers supporting single and chat interactions."""

    def complete(self, prompt: str) -> LLMResponse:
        """Generate a response for a single prompt."""
        ...

    def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        """Generate a response using chat-style message history."""
        ...

    def vision(self, prompt: str, image: ImageInput) -> LLMResponse:
        """Generate a response that includes image input."""
        ...
