"""Tests for prompt loading and conversion utilities."""

from __future__ import annotations

import pytest

from various_llm_benchmark.prompts.prompt import PromptTemplate, load_provider_prompt


def test_load_provider_prompt_reads_yaml() -> None:
    """Providers should load prompts from the matching YAML file."""
    template = load_provider_prompt("agents", "openai_agents")

    assert isinstance(template, PromptTemplate)
    assert template.system.startswith("You are an assistant that helps")

    messages = template.to_messages("hello")
    assert messages[0].role == "system"
    assert messages[-1].content == "hello"


def test_missing_prompt_raises_file_not_found() -> None:
    """Missing prompts should raise a clear error."""
    with pytest.raises(FileNotFoundError):
        load_provider_prompt("agents", "does_not_exist")
