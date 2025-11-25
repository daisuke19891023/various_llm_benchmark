"""Agno-based agent provider implementations."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Literal, Protocol, cast

from various_llm_benchmark.logger import BaseComponent
from various_llm_benchmark.models import (
    ChatMessage,
    ImageInput,
    LLMResponse,
    normalize_tool_calls,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from agno.models.anthropic import Claude
    from agno.models.google import Gemini
    from agno.models.message import Message
    from agno.models.openai import OpenAIChat

    AgentModel = OpenAIChat | Claude | Gemini
else:
    AgentModel = object


class AgentRunner(Protocol):
    """Minimal interface for running an Agno agent."""

    model: object | None

    def run(self, run_input: object, *, stream: bool | None = None, **kwargs: object) -> object:
        """Run the agent and return a result."""


class RunResult(Protocol):
    """Subset of fields returned from an Agno run."""

    content: object | None
    messages: list[Message] | None
    model: str | None
    __dict__: dict[str, object]
    __dataclass_fields__: object | None


ProviderName = Literal["openai", "anthropic", "gemini"]


class AgnoAgentProvider(BaseComponent):
    """Run Agno agents with OpenAI or Anthropic backends."""

    def __init__(
        self,
        openai_api_key: str,
        anthropic_api_key: str,
        gemini_api_key: str,
        openai_model: str,
        anthropic_model: str,
        gemini_model: str,
        *,
        temperature: float = 0.7,
        instructions: str | None = None,
        agent_factory: Callable[[AgentModel], object] | None = None,
    ) -> None:
        """Configure the provider with API keys and defaults."""
        self._openai_api_key = openai_api_key
        self._anthropic_api_key = anthropic_api_key
        self._gemini_api_key = gemini_api_key
        self._openai_model = openai_model
        self._anthropic_model = anthropic_model
        self._gemini_model = gemini_model
        self._temperature = temperature
        self._instructions = instructions
        self._agent_factory = agent_factory or self._default_agent_factory

    def complete(self, prompt: str, *, provider: ProviderName, model: str | None = None) -> LLMResponse:
        """Generate a single-turn response."""
        self.log_start("agno_complete", provider=provider, model=model or self._model_name(provider))
        self.log_io(direction="input", prompt=prompt)
        agent, model_obj = self._build_agent(provider, model)
        run_output, elapsed_seconds = self._run(agent, prompt)
        response = self._build_response(run_output, model_obj, elapsed_seconds)
        self.log_io(direction="output", model=response.model, content=response.content)
        self.log_end("agno_complete", elapsed_seconds=elapsed_seconds)
        return response

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        provider: ProviderName,
        model: str | None = None,
    ) -> LLMResponse:
        """Generate a response using message history."""
        self.log_start(
            "agno_chat",
            provider=provider,
            model=model or self._model_name(provider),
            message_count=len(messages),
        )
        if messages:
            self.log_io(
                direction="input",
                message_count=len(messages),
                last_user_message=messages[-1].content,
            )
        agent, model_obj = self._build_agent(provider, model)
        agno_messages = [self._to_agno_message(message) for message in messages]
        run_output, elapsed_seconds = self._run(agent, agno_messages)
        response = self._build_response(run_output, model_obj, elapsed_seconds)
        self.log_io(direction="output", model=response.model, content=response.content)
        self.log_end("agno_chat", elapsed_seconds=elapsed_seconds)
        return response

    def vision(
        self,
        prompt: str,
        image: ImageInput,
        *,
        provider: ProviderName,
        model: str | None = None,
    ) -> LLMResponse:
        """Generate a response using an image input."""
        self.log_start(
            "agno_vision",
            provider=provider,
            model=model or self._model_name(provider),
            image_media_type=image.media_type,
        )
        self.log_io(direction="input", prompt=prompt, image_bytes=len(image.data))
        agent, model_obj = self._build_agent(provider, model)
        run_input = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": image.as_data_url()},
                ],
            },
        ]
        run_output, elapsed_seconds = self._run(agent, run_input)
        response = self._build_response(run_output, model_obj, elapsed_seconds)
        self.log_io(direction="output", model=response.model, content=response.content)
        self.log_end("agno_vision", elapsed_seconds=elapsed_seconds)
        return response

    def _model_name(self, provider: ProviderName) -> str:
        if provider == "openai":
            return self._openai_model
        if provider == "anthropic":
            return self._anthropic_model
        return self._gemini_model

    def _build_agent(
        self,
        provider: ProviderName,
        model: str | None,
    ) -> tuple[AgentRunner, AgentModel]:
        model_obj = self._create_model(provider, model)
        runner = cast("AgentRunner", self._agent_factory(model_obj))
        return runner, model_obj

    def _create_model(self, provider: ProviderName, model: str | None) -> AgentModel:
        if provider == "openai":
            from agno.models.openai import OpenAIChat

            return OpenAIChat(
                id=model or self._openai_model,
                temperature=self._temperature,
                api_key=self._openai_api_key,
            )
        if provider == "anthropic":
            from agno.models.anthropic import Claude

            return Claude(
                id=model or self._anthropic_model,
                temperature=self._temperature,
                api_key=self._anthropic_api_key,
            )
        from agno.models.google import Gemini

        return Gemini(
            id=model or self._gemini_model,
            temperature=self._temperature,
            api_key=self._gemini_api_key,
        )

    @staticmethod
    def _to_agno_message(message: ChatMessage) -> Message:
        from agno.models.message import Message

        return Message(role=message.role, content=message.content)

    def _run(self, agent: AgentRunner, run_input: object) -> tuple[RunResult, float]:
        start = perf_counter()
        run_output = cast("RunResult", agent.run(run_input, stream=False))
        elapsed_seconds = perf_counter() - start
        return run_output, elapsed_seconds

    def _build_response(self, run_output: RunResult, model: AgentModel, elapsed_seconds: float) -> LLMResponse:
        content = self._extract_content(run_output)
        model_name = run_output.model or model.id
        raw_output = run_output.__dict__.copy()
        tool_calls = normalize_tool_calls(raw_output)
        return LLMResponse(
            content=content,
            model=model_name,
            raw=raw_output,
            elapsed_seconds=elapsed_seconds,
            call_count=1,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _extract_content(run_output: RunResult) -> str:
        if isinstance(run_output.content, str):
            return run_output.content
        if run_output.messages:
            last_message = run_output.messages[-1]
            content = last_message.get_content_string()
            if content:
                return content
        return str(run_output.content)

    def _default_agent_factory(self, model: AgentModel) -> AgentRunner:
        from agno.agent import Agent

        return cast("AgentRunner", Agent(model=model, instructions=self._instructions))
