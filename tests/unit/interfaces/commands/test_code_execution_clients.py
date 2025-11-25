"""Tests for provider-specific code execution callers."""

from typing import Any, cast

from various_llm_benchmark.interfaces.commands.code_execution_clients import (
    AnthropicCodeExecutionTool,
    GeminiCodeExecutionTool,
    OpenAICodeExecutionTool,
)


class _StubOpenAIResponse:
    def __init__(self, *, model: str = "gpt-5.1", output: object = None) -> None:
        self.model = model
        self.output = output or {"content": [{"text": {"value": "4"}}]}

    def model_dump(self) -> dict[str, object]:
        return {
            "model": self.model,
            "output": self.output,
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "code-execute",
                    "output": "4",
                    "type": "function",
                },
            ],
            "usage": {"completion_tokens": 10},
        }


class _StubResponsesClient:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None
        self._response = _StubOpenAIResponse()

    def create(self, **kwargs: object) -> _StubOpenAIResponse:
        self.kwargs = kwargs
        return self._response


class _StubOpenAIClient:
    def __init__(self) -> None:
        self.responses = _StubResponsesClient()


class _StubAnthropicResponse:
    def __init__(self, *, model: str = "claude-sonnet", content: object = None) -> None:
        self.model = model
        self.content = content or [{"text": "6"}]

    def model_dump(self) -> dict[str, object]:
        return {
            "model": self.model,
            "content": self.content,
            "tool_calls": [
                {
                    "id": "code_call_1",
                    "name": "code_execution",
                    "output": "6",
                },
            ],
        }


class _StubMessagesClient:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None
        self._response = _StubAnthropicResponse()

    def create(self, **kwargs: object) -> _StubAnthropicResponse:
        self.kwargs = kwargs
        return self._response


class _StubAnthropicClient:
    def __init__(self) -> None:
        self.messages = _StubMessagesClient()


class _StubGeminiResponse:
    def __init__(self, *, model: str = "gemini-3.0-pro", text: str = "9") -> None:
        self.model = model
        self.text = text

    def model_dump(self) -> dict[str, object]:
        return {
            "model": self.model,
            "text": self.text,
            "tool_calls": [{"name": "code_execution", "output": self.text}],
        }


class _StubGeminiModels:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None
        self._response = _StubGeminiResponse()

    def generate_content(self, **kwargs: object) -> _StubGeminiResponse:
        self.kwargs = kwargs
        return self._response


class _StubGeminiClient:
    def __init__(self) -> None:
        self.models = _StubGeminiModels()


def test_openai_code_execution_invokes_code_interpreter() -> None:
    """OpenAI caller should request the code interpreter tool and parse output."""
    client = _StubOpenAIClient()
    caller = OpenAICodeExecutionTool(cast("Any", client), "gpt-5.1", temperature=0.2, system_prompt="calc")

    result = caller.run("2+2", model="gpt-5.1-mini")

    kwargs = client.responses.kwargs
    assert kwargs is not None
    assert kwargs["tools"] == [{"type": "code_interpreter"}]
    assert kwargs["tool_choice"] == "auto"
    assert cast("list[dict[str, str]]", kwargs["input"])[0]["role"] == "system"
    assert result.content == "4"
    assert result.tool_calls[0].output == "4"


def test_anthropic_code_execution_invokes_tool_call() -> None:
    """Claude caller should include the Anthropic code execution tool payload."""
    client = _StubAnthropicClient()
    caller = AnthropicCodeExecutionTool(cast("Any", client), "claude-sonnet", temperature=0.1, system_prompt="calc")

    result = caller.run("3*2", model="claude-haiku")

    assert client.messages.kwargs is not None
    assert client.messages.kwargs["tools"] == [
        {"type": "code_execution_20250825", "name": "code_execution"},
    ]
    assert client.messages.kwargs["tool_choice"] == {"type": "auto"}
    assert client.messages.kwargs["system"] == "calc"
    assert result.content == "6"
    assert result.tool_calls[0].name == "code_execution"


def test_gemini_code_execution_configures_tools() -> None:
    """Gemini caller should send tool configuration and parse text output."""
    client = _StubGeminiClient()
    caller = GeminiCodeExecutionTool(cast("Any", client), "gemini-3.0-pro", temperature=0.3, system_prompt="calc")

    result = caller.run("10-1", model="gemini-2.5-flash")

    assert client.models.kwargs is not None
    assert client.models.kwargs["tools"] == [{"code_execution": {}}]
    assert client.models.kwargs["tool_config"] == {"code_execution": {"mode": "AUTO"}}
    assert client.models.kwargs["system_instruction"] == "calc"
    assert result.content == "9"
    assert result.tool_calls[0].name == "code_execution"
