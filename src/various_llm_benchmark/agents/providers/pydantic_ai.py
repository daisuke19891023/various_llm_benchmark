"""Pydantic AI agent provider wrapper."""

from __future__ import annotations

import dataclasses
import inspect
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pydantic_ai import RunContext
    from pydantic_ai.messages import (
        ModelMessage,
        UserContent,
    )
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import Tool

    from various_llm_benchmark.llm.tools.registry import ToolRegistration

from various_llm_benchmark.logger import BaseComponent
from various_llm_benchmark.models import (
    ChatMessage,
    ImageInput,
    LLMResponse,
    normalize_tool_calls,
)


class AgentRunner(Protocol):
    """Protocol for objects capable of running Pydantic AI agents."""

    def run(self, *args: object, **kwargs: object) -> object:
        """Run the agent and return a provider-specific result."""


class AgentRunResult(Protocol):
    """Subset of an agent run result used for normalization."""

    response: object
    all_messages: Sequence[ModelMessage] | None
    new_messages: Sequence[ModelMessage] | None
    run_id: str | None


class PydanticAIAgentProvider(BaseComponent):
    """Wrapper around pydantic-ai's Agent for unified interface."""

    def __init__(
        self,
        *,
        model: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        tools: list[ToolRegistration] | None = None,
        agent_factory: Callable[..., AgentRunner] | None = None,
    ) -> None:
        """Store defaults for later calls."""
        self._model = model
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._tools = tools or []
        self._agent_factory = agent_factory or self._default_agent_factory

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        tools: list[ToolRegistration] | None = None,
    ) -> LLMResponse:
        """Generate a single-turn response."""
        self.log_start("pydantic_ai_complete", model=model or self._model)
        self.log_io(direction="input", prompt=prompt)
        agent = self._build_agent(
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            tools=tools,
        )
        run_result, elapsed_seconds = self._run(agent, prompt)
        response = self._build_response(run_result, model or self._model, elapsed_seconds)
        self.log_io(direction="output", model=response.model, content=response.content)
        self.log_end("pydantic_ai_complete", elapsed_seconds=elapsed_seconds)
        return response

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        tools: list[ToolRegistration] | None = None,
    ) -> LLMResponse:
        """Generate a response using prior chat history."""
        model_name = model or self._model
        self.log_start("pydantic_ai_chat", model=model_name, message_count=len(messages))
        if messages:
            self.log_io(
                direction="input",
                message_count=len(messages),
                last_user_message=messages[-1].content,
            )
        agent = self._build_agent(
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            tools=tools,
        )
        history = [self._to_model_message(message) for message in messages]
        run_result, elapsed_seconds = self._run(agent, None, message_history=history)
        response = self._build_response(run_result, model or self._model, elapsed_seconds)
        self.log_io(direction="output", model=response.model, content=response.content)
        self.log_end("pydantic_ai_chat", elapsed_seconds=elapsed_seconds)
        return response

    def vision(
        self,
        prompt: str,
        image: ImageInput,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        tools: list[ToolRegistration] | None = None,
    ) -> LLMResponse:
        """Generate a response that includes image content."""
        model_name = model or self._model
        self.log_start(
            "pydantic_ai_vision",
            model=model_name,
            image_media_type=image.media_type,
        )
        self.log_io(direction="input", prompt=prompt, image_bytes=len(image.data))
        agent = self._build_agent(
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            tools=tools,
        )
        from pydantic_ai.messages import ImageUrl

        image_prompt: list[UserContent] = [prompt, ImageUrl(url=image.as_data_url(), media_type=image.media_type)]
        run_result, elapsed_seconds = self._run(agent, image_prompt)
        response = self._build_response(run_result, model or self._model, elapsed_seconds)
        self.log_io(direction="output", model=response.model, content=response.content)
        self.log_end("pydantic_ai_vision", elapsed_seconds=elapsed_seconds)
        return response

    def _build_agent(
        self,
        *,
        system_prompt: str | None,
        model: str | None,
        temperature: float | None,
        tools: list[ToolRegistration] | None,
    ) -> AgentRunner:
        agent_tools = self._convert_tools(tools or self._tools)
        return self._agent_factory(
            model=model or self._model,
            instructions=system_prompt or self._system_prompt,
            model_settings=self._build_model_settings(temperature),
            tools=agent_tools,
        )

    def _build_model_settings(self, temperature: float | None) -> ModelSettings:
        from pydantic_ai.settings import ModelSettings

        model_temperature = temperature if temperature is not None else self._temperature
        return ModelSettings(temperature=model_temperature)

    @staticmethod
    def _to_model_message(message: ChatMessage) -> ModelMessage:
        from pydantic_ai.messages import ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart

        if message.role == "assistant":
            return ModelResponse(parts=[TextPart(content=message.content)])
        if message.role == "system":
            return ModelRequest(parts=[SystemPromptPart(content=message.content)])
        return ModelRequest(parts=[UserPromptPart(content=message.content)])

    def _convert_tools(self, registrations: list[ToolRegistration]) -> list[Tool]:
        return [self._wrap_tool(registration) for registration in registrations]

    def _wrap_tool(self, registration: ToolRegistration) -> Tool:
        from pydantic_ai.tools import Tool

        handler = registration.handler
        takes_ctx = self._handler_accepts_run_context(handler)

        def tool_fn(run_context: RunContext, **kwargs: Any) -> object:
            if takes_ctx:
                return handler(run_context, **kwargs)
            return handler(**kwargs)

        return Tool(tool_fn, name=registration.name, description=registration.description)

    def _handler_accepts_run_context(self, handler: Callable[..., object]) -> bool:
        from pydantic_ai import RunContext

        signature = inspect.signature(handler)
        parameters = list(signature.parameters.values())
        if not parameters:
            return False
        first = parameters[0]
        if first.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            return False
        if first.annotation is RunContext:
            return True
        return first.name in {"run_context", "ctx", "context"}

    def _run(self, agent: AgentRunner, *args: object, **kwargs: object) -> tuple[AgentRunResult, float]:
        start = perf_counter()
        run_result = cast("AgentRunResult", agent.run(*args, **kwargs))
        elapsed_seconds = perf_counter() - start
        return run_result, elapsed_seconds

    def _build_response(self, run_result: AgentRunResult, model_name: str, elapsed_seconds: float) -> LLMResponse:
        content = self._extract_content(run_result)
        raw_output = self._dump_raw(run_result)
        tool_calls = normalize_tool_calls(raw_output)
        return LLMResponse(
            content=content,
            model=model_name,
            raw=raw_output,
            elapsed_seconds=elapsed_seconds,
            call_count=1,
            tool_calls=tool_calls,
        )

    def _extract_content(self, run_result: AgentRunResult) -> str:
        response = run_result.response
        if isinstance(response, str):
            return response

        messages = run_result.all_messages or run_result.new_messages or []
        text = self._extract_text(messages)
        if text:
            return text
        return str(response)

    def _extract_text(self, messages: Sequence[ModelMessage]) -> str:
        from pydantic_ai.messages import ModelResponse, TextPart

        for message in reversed(messages):
            if isinstance(message, ModelResponse):
                parts = getattr(message, "parts", [])
                texts = [part.content for part in parts if isinstance(part, TextPart)]
                if texts:
                    return "\n".join(texts)
        return ""

    def _dump_raw(self, run_result: AgentRunResult) -> dict[str, object]:
        if dataclasses.is_dataclass(run_result):
            try:
                return dataclasses.asdict(run_result)
            except TypeError:
                pass
        dictionary = getattr(run_result, "__dict__", None)
        if dictionary is not None:
            return dict(dictionary)

        raw: dict[str, object] = {}
        for field in ("response", "all_messages", "new_messages", "run_id"):
            if hasattr(run_result, field):
                raw[field] = getattr(run_result, field)
        return raw

    def _default_agent_factory(
        self,
        *,
        model: str,
        instructions: str | None,
        model_settings: ModelSettings,
        tools: Sequence[Tool],
    ) -> AgentRunner:
        from pydantic_ai import Agent

        return cast(
            "AgentRunner",
            Agent(
                model=model,
                instructions=instructions,
                model_settings=model_settings,
                tools=tools,
            ),
        )
