from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel, Field

from various_llm_benchmark.models import ChatMessage

if TYPE_CHECKING:
    from collections.abc import Sequence

PromptCategory = Literal["agents", "llm"]


class PromptTemplate(BaseModel):
    """Prompt definition loaded from YAML."""

    system: str = Field(..., description="System prompt content")

    def to_system_message(self) -> ChatMessage:
        """Return the system prompt as a chat message."""
        return ChatMessage(role="system", content=self.system)

    def to_prompt_text(self, prompt: str) -> str:
        """Combine the system instructions and a user prompt into a single string."""
        return f"{self.system}\n\n{prompt}"

    def to_messages(self, prompt: str, history: Sequence[ChatMessage] | None = None) -> list[ChatMessage]:
        """Combine the system instructions, existing history, and the user prompt."""
        messages: list[ChatMessage] = [self.to_system_message()]
        messages.extend(history or [])
        messages.append(ChatMessage(role="user", content=prompt))
        return messages


def load_provider_prompt(category: PromptCategory, provider: str) -> PromptTemplate:
    """Load a provider-specific prompt from the prompts directory."""
    filename = provider if provider.endswith(".yaml") else f"{provider}.yaml"
    base_path = resources.files("various_llm_benchmark.prompts")
    prompt_path = base_path.joinpath(category, "providers", filename)
    if not prompt_path.is_file():
        missing_path = Path(category) / "providers" / filename
        error_message = f"Prompt template not found: {missing_path}"
        raise FileNotFoundError(error_message)

    content = prompt_path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    if data is None:
        error_message = f"Prompt template is empty: {prompt_path}"
        raise ValueError(error_message)

    return PromptTemplate.model_validate(data)
