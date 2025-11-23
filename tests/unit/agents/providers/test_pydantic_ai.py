from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

import various_llm_benchmark.agents.providers.pydantic_ai as pydantic_ai_module

from pydantic_ai.messages import (
    ImageUrl,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
)
from pydantic_ai.tools import Tool
from various_llm_benchmark.agents.providers.pydantic_ai import PydanticAIAgentProvider
from various_llm_benchmark.llm.tools.registry import ToolCategory, ToolRegistration
from various_llm_benchmark.models import ChatMessage, ImageInput



class RunRecord(TypedDict):
    """Structure for recorded agent run calls."""

    user_prompt: object | None
    kwargs: dict[str, object]


def _temperature_value(model_settings: Any) -> float | None:
    """Extract temperature value from model settings objects or dicts."""
    if hasattr(model_settings, "temperature"):
        return cast("float | None", model_settings.temperature)
    if isinstance(model_settings, dict):
        settings_dict = cast("dict[str, object]", model_settings)
        return cast("float | None", settings_dict.get("temperature"))
    return None


def _empty_object_list() -> list[object]:
    """Return an empty list of objects for dataclass defaults."""
    return []


@dataclass
class StubRunResult:
    """Simple stub to emulate AgentRunResult fields."""

    response: object
    all_messages: list[object] = field(default_factory=_empty_object_list)
    new_messages: list[object] = field(default_factory=_empty_object_list)
    run_id: str | None = None


class StubAgent:
    """Stub agent capturing arguments sent to run."""

    def __init__(self, **kwargs: object) -> None:
        """Store provided configuration for inspection."""
        self.kwargs: dict[str, object] = dict(kwargs)
        self.runs: list[RunRecord] = []

    def run(self, *args: object, **kwargs: object) -> StubRunResult:
        """Record invocations for assertions."""
        user_prompt = args[0] if args else kwargs.get("user_prompt")
        run_kwargs: RunRecord = {"user_prompt": user_prompt, "kwargs": dict(kwargs)}
        self.runs.append(run_kwargs)
        return StubRunResult(response="ok", all_messages=[])


class StubToolAgent(StubAgent):
    """Agent variant that triggers tool execution inside run."""

    def run(self, *args: object, **kwargs: object) -> StubRunResult:
        """Run tools when present and record the invocation."""
        user_prompt = args[0] if args else kwargs.get("user_prompt")
        run_kwargs: RunRecord = {"user_prompt": user_prompt, "kwargs": dict(kwargs)}
        tools: list[Tool] = list(cast("list[Tool]", self.kwargs.get("tools", [])))
        tool = cast("Any", tools[0]) if tools else None
        tool_callable = getattr(tool, "function", None) if tool is not None else None
        tool_response = (
            cast("str", tool_callable(object(), action="ping")) if callable(tool_callable) else "no-tools"
        )
        self.runs.append(run_kwargs)
        return StubRunResult(response=tool_response)


def test_complete_uses_defaults_and_passes_prompt(monkeypatch: Any) -> None:
    """Complete should forward prompt and defaults to the agent."""
    created: list[StubAgent] = []
    times = iter([0.0, 0.75])
    monkeypatch.setattr(pydantic_ai_module, "perf_counter", lambda: next(times))

    def agent_factory(**kwargs: object) -> StubAgent:
        agent = StubAgent(**kwargs)
        created.append(agent)
        return agent

    provider = PydanticAIAgentProvider(
        model="gpt-mini",
        system_prompt="sys",
        temperature=0.2,
        agent_factory=agent_factory,
    )

    response = provider.complete("hello")

    assert response.content == "ok"
    assert response.elapsed_seconds == 0.75
    assert response.call_count == 1
    assert response.tool_calls == []
    assert created
    agent = created[0]
    assert agent.kwargs["model"] == "gpt-mini"
    assert agent.kwargs["instructions"] == "sys"
    model_settings = agent.kwargs["model_settings"]
    temperature = _temperature_value(model_settings)
    assert temperature == 0.2
    run_call = agent.runs[0]
    assert run_call["user_prompt"] == "hello"


def test_chat_converts_history_to_model_messages() -> None:
    """Chat should map ChatMessage history into ModelMessages."""
    created: list[StubAgent] = []

    def agent_factory(**kwargs: object) -> StubAgent:
        agent = StubAgent(**kwargs)
        created.append(agent)
        return agent

    provider = PydanticAIAgentProvider(
        model="gpt-mini",
        agent_factory=agent_factory,
    )
    history = [
        ChatMessage(role="system", content="context"),
        ChatMessage(role="user", content="question"),
        ChatMessage(role="assistant", content="answer"),
    ]

    response = provider.chat(history, model="other")

    assert response.content == "ok"
    agent = created[0]
    run_kwargs: dict[str, object] = agent.runs[0]["kwargs"]
    message_history = cast("list[ModelRequest | ModelResponse]", run_kwargs["message_history"])
    assert isinstance(message_history, list)
    assert isinstance(message_history[0], ModelRequest)
    assert isinstance(message_history[1], ModelRequest)
    assert isinstance(message_history[2], ModelResponse)
    system_part = message_history[0].parts[0]
    assert isinstance(system_part, SystemPromptPart)
    assert system_part.content == "context"
    assistant_text = message_history[2].parts[0]
    assert isinstance(assistant_text, TextPart)
    assert assistant_text.content == "answer"


def test_vision_encodes_image_input() -> None:
    """Vision should attach prompt and encoded image to user prompt list."""
    created: list[StubAgent] = []

    def agent_factory(**kwargs: object) -> StubAgent:
        agent = StubAgent(**kwargs)
        created.append(agent)
        return agent

    provider = PydanticAIAgentProvider(
        model="gpt-mini",
        agent_factory=agent_factory,
    )
    image = ImageInput(media_type="image/png", data="ZGF0YQ==")

    response = provider.vision("see", image, temperature=0.9)

    assert response.content == "ok"
    agent = created[0]
    run_call: RunRecord = agent.runs[0]
    prompt_input = cast("list[object]", run_call["user_prompt"])
    assert isinstance(prompt_input, list)
    assert prompt_input[0] == "see"
    assert isinstance(prompt_input[1], ImageUrl)
    assert prompt_input[1].url.endswith(image.data)
    model_settings = agent.kwargs["model_settings"]
    vision_temperature = _temperature_value(model_settings)
    assert vision_temperature == 0.9


def test_tools_are_wrapped_and_callable_with_context() -> None:
    """Registered tools should be transformed into pydantic-ai Tool instances."""
    tool_calls: list[dict[str, object]] = []

    def handler(ctx: object, action: str) -> str:
        tool_calls.append({"ctx": ctx, "action": action})
        return action.upper()

    registration = ToolRegistration(
        id="echo",
        name="echo",
        description="Echo tool",
        input_schema={"type": "object", "properties": {"action": {"type": "string"}}},
        handler=handler,
        category=ToolCategory.EXTERNAL,
    )

    created: list[StubToolAgent] = []

    def agent_factory(**kwargs: object) -> StubToolAgent:
        agent = StubToolAgent(**kwargs)
        created.append(agent)
        return agent

    provider = PydanticAIAgentProvider(
        model="gpt-mini",
        tools=[registration],
        agent_factory=agent_factory,
    )

    response = provider.complete("do something")

    assert response.content == "PING"
    assert created
    agent = created[0]
    tool_list = cast("list[Tool]", agent.kwargs["tools"])
    assert isinstance(tool_list, list)
    assert isinstance(tool_list[0], Tool)
    assert tool_calls[0]["action"] == "ping"
    assert tool_calls[0]["ctx"]
