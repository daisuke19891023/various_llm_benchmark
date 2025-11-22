"""Shared Pydantic data models used across LLM and agent providers."""

from __future__ import annotations

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """Represents a single chat message."""

    role: str
    content: str


class ImageInput(BaseModel):
    """Represents an image payload encoded for LLM consumption."""

    media_type: str
    data: str

    def as_data_url(self) -> str:
        """Return the image as a data URL for providers that expect URLs."""
        return f"data:{self.media_type};base64,{self.data}"


class LLMResponse(BaseModel):
    """Normalized response from a provider."""

    content: str
    model: str
    raw: dict[str, object]
