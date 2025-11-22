"""Tests for web search tool wrappers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from various_llm_benchmark.llm.tools.web_search import (
    AnthropicWebSearchTool,
    GeminiWebSearchTool,
    OpenAIWebSearchTool,
    SupportsMessages,
    SupportsResponses,
    SupportsSearchModels,
)


class FakeResponses:
    """Stub for the OpenAI responses client."""

    def __init__(self) -> None:
        """Initialize capture storage."""
        self.kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> SimpleNamespace:
        """Record the request and return a fake response object."""
        self.kwargs = kwargs
        model = kwargs.get("model", "")
        output = "search-result"
        return SimpleNamespace(
            model=model,
            output=output,
            model_dump=lambda: {"model": model, "input": output},
        )


class FakeMessages:
    """Stub for the Anthropic messages client."""

    def __init__(self) -> None:
        """Initialize capture storage."""
        self.kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> SimpleNamespace:
        """Record the request and return a fake response object."""
        self.kwargs = kwargs
        model = kwargs.get("model", "")
        content = [{"text": "answer"}]
        return SimpleNamespace(
            model=model,
            content=content,
            model_dump=lambda: {"model": model, "content": content},
        )


class FakeOpenAIClient:
    """Client stub that exposes a responses attribute."""

    def __init__(self, responses: FakeResponses) -> None:
        """Store the responses stub."""
        self.responses = responses


class FakeAnthropicClient:
    """Client stub that exposes a messages attribute."""

    def __init__(self, messages: FakeMessages) -> None:
        """Store the messages stub."""
        self.messages = messages


class FakeGeminiModels:
    """Stub for the Gemini models client."""

    def __init__(self) -> None:
        """Initialize capture storage."""
        self.kwargs: dict[str, object] = {}

    def generate_content(self, **kwargs: object) -> SimpleNamespace:
        """Record the request and return a fake response object."""
        self.kwargs = kwargs
        model = kwargs.get("model", "")
        text = "web" if isinstance(model, str) else ""
        return SimpleNamespace(text=text, model=model, model_dump=lambda: {"model": model, "tool": "web"})


class FakeGeminiClient:
    """Client stub that exposes a models attribute."""

    def __init__(self, models: FakeGeminiModels) -> None:
        """Store the models stub."""
        self.models = models


def test_openai_web_search_builds_request() -> None:
    """OpenAI web search should include system prompt and tools."""
    responses = FakeResponses()
    client = cast("SupportsResponses", FakeOpenAIClient(responses))
    tool = OpenAIWebSearchTool(client, "gpt-default", temperature=0.3, system_prompt="sys")

    result = tool.search("hello", model="gpt-4")

    assert result.content == "search-result"
    assert result.model == "gpt-4"
    assert responses.kwargs["tools"] == [{"type": "web_search"}]
    assert responses.kwargs["tool_choice"] == "auto"
    assert responses.kwargs["temperature"] == 0.3
    inputs = cast("list[dict[str, str]]", responses.kwargs["input"])
    assert inputs[0]["content"] == "sys"


def test_anthropic_web_search_builds_request() -> None:
    """Anthropic web search should forward tool parameters."""
    messages = FakeMessages()
    client = cast("SupportsMessages", FakeAnthropicClient(messages))
    tool = AnthropicWebSearchTool(client, "claude-default", temperature=0.2, system_prompt="sys")

    result = tool.search("question")

    assert result.content == "answer"
    assert result.model == "claude-default"
    assert messages.kwargs["system"] == "sys"
    assert messages.kwargs["tools"] == [{"type": "web_search"}]
    assert messages.kwargs["tool_choice"] == {"type": "auto"}
    assert messages.kwargs["temperature"] == 0.2


def test_anthropic_web_search_accepts_model_override() -> None:
    """Model override should propagate to Anthropic web search requests."""
    messages = FakeMessages()
    client = cast("SupportsMessages", FakeAnthropicClient(messages))
    tool = AnthropicWebSearchTool(client, "claude-default")

    result = tool.search("question", model="claude-3")

    assert result.model == "claude-3"
    assert messages.kwargs["model"] == "claude-3"


def test_gemini_web_search_builds_request() -> None:
    """Gemini web search should send tool config and system prompt."""
    models = FakeGeminiModels()
    client = cast("SupportsSearchModels", FakeGeminiClient(models))
    tool = GeminiWebSearchTool(client, "gemini-default", temperature=0.6, system_prompt="sys")

    result = tool.search("find news", model="gemini-2.0")

    assert result.content == "web"
    assert result.model == "gemini-2.0"
    assert models.kwargs["system_instruction"] == "sys"
    assert models.kwargs["tools"] == [{"google_search_retrieval": {}}]
    assert models.kwargs["tool_config"] == {
        "google_search_retrieval": {"dynamic_retrieval_config": {"mode": "DYNAMIC"}},
    }
    assert models.kwargs["config"] == {"temperature": 0.6}
