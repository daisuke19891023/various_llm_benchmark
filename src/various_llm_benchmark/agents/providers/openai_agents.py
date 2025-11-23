from __future__ import annotations

import asyncio
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol, cast

from agents import Agent, set_default_openai_key
from agents.items import ItemHelpers, TResponseInputItem
from agents.model_settings import ModelSettings
from agents.run import Runner

from various_llm_benchmark.models import (
    ChatMessage,
    ImageInput,
    LLMResponse,
    normalize_tool_calls,
)
from various_llm_benchmark.llm.tools.payloads import to_agents_sdk_tools_payload

if TYPE_CHECKING:
    from agents.result import RunResult
    from various_llm_benchmark.llm.tools.registry import ToolRegistration


class AgentRunFunction(Protocol):
    """Callable that executes an Agents SDK run synchronously."""

    def __call__(self, agent: Agent, run_input: str | list[TResponseInputItem]) -> RunResult:
        """Execute the agent with the given input and return the run result."""
        ...


class OpenAIAgentsProvider:
    """Wrapper around the OpenAI Agents SDK for simple completions and chats."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        instructions: str,
        temperature: float = 0.7,
        run_function: AgentRunFunction | None = None,
        tools: list[ToolRegistration] | None = None,
    ) -> None:
        """Configure the provider and default agent settings."""
        set_default_openai_key(api_key, use_for_tracing=False)
        self._model = model
        self._instructions = instructions
        self._temperature = temperature
        self._run_function = run_function or self._default_run_function
        self._tools = tools or []

    def complete(self, prompt: str) -> LLMResponse:
        """Generate a single-turn response via Agents SDK."""
        agent = self._build_agent()
        run_result, elapsed_seconds = self._run(agent, prompt)
        return self._to_response(run_result, elapsed_seconds)

    def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        """Generate a response using chat-style history."""
        agent = self._build_agent()
        run_input: list[TResponseInputItem] = [self._to_agent_message(message) for message in messages]
        run_result, elapsed_seconds = self._run(agent, run_input)
        return self._to_response(run_result, elapsed_seconds)

    def vision(self, prompt: str, image: ImageInput) -> LLMResponse:
        """Generate a response that includes image content."""
        agent = self._build_agent()
        run_input: list[TResponseInputItem] = [
            cast(
                "TResponseInputItem",
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image.as_data_url()},
                    ],
                },
            ),
        ]
        run_result, elapsed_seconds = self._run(agent, run_input)
        return self._to_response(run_result, elapsed_seconds)

    def _build_agent(self) -> Agent:
        agent_tools = to_agents_sdk_tools_payload(self._tools) if self._tools else []
        typed_tools = cast("list[Any]", agent_tools)
        return Agent(
            name="openai-agents-sdk",
            instructions=self._instructions,
            model=self._model,
            model_settings=ModelSettings(temperature=self._temperature),
            tools=typed_tools,
        )

    @staticmethod
    def _to_agent_message(message: ChatMessage) -> TResponseInputItem:
        return cast("TResponseInputItem", {"role": message.role, "content": message.content})

    def _run(self, agent: Agent, run_input: str | list[TResponseInputItem]) -> tuple[RunResult, float]:
        start = perf_counter()
        result = self._run_function(agent, run_input)
        elapsed_seconds = perf_counter() - start
        return result, elapsed_seconds

    def _to_response(self, run_result: RunResult, elapsed_seconds: float) -> LLMResponse:
        content = self._extract_content(run_result)
        raw_output = run_result.__dict__.copy()
        tool_calls = normalize_tool_calls(raw_output)
        return LLMResponse(
            content=content,
            model=self._model,
            raw=raw_output,
            elapsed_seconds=elapsed_seconds,
            call_count=1,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _extract_content(run_result: RunResult) -> str:
        if isinstance(run_result.final_output, str):
            return run_result.final_output
        text = ItemHelpers.text_message_outputs(run_result.new_items)
        if text:
            return text
        return str(run_result.final_output)

    @staticmethod
    def _default_run_function(agent: Agent, run_input: str | list[TResponseInputItem]) -> RunResult:
        return asyncio.run(Runner.run(starting_agent=agent, input=run_input))
