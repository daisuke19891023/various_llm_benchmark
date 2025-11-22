"""Protocols for LLM providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from various_llm_benchmark.models import ChatMessage, ImageInput, LLMResponse


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM providers that support text and chat generation."""

    def generate(self, prompt: str, *, model: str | None = None) -> LLMResponse:
        """Generate a response for a single prompt."""
        ...

    def chat(self, messages: list[ChatMessage], *, model: str | None = None) -> LLMResponse:
        """Generate a response using a chat-style message history."""
        ...

    def vision(
        self,
        prompt: str,
        image: ImageInput,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a response that combines text with an image input."""
        ...
