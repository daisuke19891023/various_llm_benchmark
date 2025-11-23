from __future__ import annotations

import asyncio
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from uuid import uuid4

from google.adk import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.runners import InMemoryRunner
from google.genai import types

from various_llm_benchmark.logger import BaseComponent
from various_llm_benchmark.models import (
    ChatMessage,
    ImageInput,
    LLMResponse,
    normalize_tool_calls,
)

if TYPE_CHECKING:
    from google.adk.sessions.session import Session

RunnerFactory = Callable[[Agent], object]
RunFunction = Callable[[Any, str, types.Content, RunConfig], Iterable[Event]]


class GoogleADKProvider(BaseComponent):
    """Wrapper around Google ADK for simple text, chat, and vision calls."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        instructions: str,
        temperature: float = 0.7,
        runner_factory: RunnerFactory | None = None,
        run_function: RunFunction | None = None,
    ) -> None:
        """Configure provider defaults and runner hooks."""
        self._api_key = api_key
        self._model = model
        self._instructions = instructions
        self._temperature = temperature
        self._runner_factory = runner_factory or self._default_runner_factory
        self._run_function = run_function or self._default_run_function
        self._run_config = RunConfig()
        self._user_id = "user"
        self._agent_name = "google_adk_agent"
        self._run_metadata = {"temperature": self._temperature}

    def complete(self, prompt: str) -> LLMResponse:
        """Generate a single-turn response."""
        self.log_start("google_adk_complete", model=self._model)
        self.log_io(direction="input", prompt=prompt)
        agent = self._build_agent()
        runner = cast("Any", self._runner_factory(agent))
        session = self._create_session(runner)
        new_message = self._to_content(ChatMessage(role="user", content=prompt))
        events, elapsed_seconds = self._run(runner, session.id, new_message)
        result = self._to_response(events, elapsed_seconds)
        self.log_io(direction="output", model=result.model, content=result.content)
        self.log_end("google_adk_complete", elapsed_seconds=elapsed_seconds)
        return result

    def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        """Generate a response with chat-style history."""
        self.log_start("google_adk_chat", model=self._model, message_count=len(messages))
        agent = self._build_agent()
        runner = cast("Any", self._runner_factory(agent))
        session = self._create_session(runner)
        history = messages[:-1]
        if history:
            session_service = cast("Any", getattr(runner, "session_service", None))
            self._seed_history(session_service, session, history)
        latest = messages[-1]
        new_message = self._to_content(latest)
        events, elapsed_seconds = self._run(runner, session.id, new_message)
        result = self._to_response(events, elapsed_seconds)
        self.log_io(direction="output", model=result.model, content=result.content)
        self.log_end("google_adk_chat", elapsed_seconds=elapsed_seconds)
        return result

    def vision(self, prompt: str, image: ImageInput) -> LLMResponse:
        """Generate a response that includes image input."""
        self.log_start("google_adk_vision", model=self._model, image_media_type=image.media_type)
        self.log_io(direction="input", prompt=prompt, image_bytes=len(image.data))
        agent = self._build_agent()
        runner = cast("Any", self._runner_factory(agent))
        session = self._create_session(runner)
        parts = [
            types.Part.from_text(text=prompt),
            types.Part.from_uri(file_uri=image.as_data_url(), mime_type=image.media_type),
        ]
        new_message = types.Content(role="user", parts=parts)
        events, elapsed_seconds = self._run(runner, session.id, new_message)
        result = self._to_response(events, elapsed_seconds)
        self.log_io(direction="output", model=result.model, content=result.content)
        self.log_end("google_adk_vision", elapsed_seconds=elapsed_seconds)
        return result

    def _build_agent(self) -> Agent:
        return Agent(
            name=self._agent_name,
            description="Agent powered by Google ADK",
            model=self._model,
            instruction=self._instructions,
        )

    def _create_session(self, runner: Any) -> Session:
        session_service = cast("Any", getattr(runner, "session_service", None))
        if session_service is None:
            error_message = "Runner is missing session_service"
            raise ValueError(error_message)
        session_id = str(uuid4())
        sync_creator = getattr(session_service, "create_session_sync", None)
        if callable(sync_creator):
            return cast(
                "Session",
                sync_creator(app_name=runner.app_name, user_id=self._user_id, session_id=session_id),
            )
        async_creator = cast("Callable[..., Awaitable[Session]]", session_service.create_session)
        awaitable = async_creator(app_name=runner.app_name, user_id=self._user_id, session_id=session_id)
        return asyncio.run(cast("Coroutine[Any, Any, Session]", awaitable))

    def _seed_history(self, session_service: Any, session: Session, history: list[ChatMessage]) -> None:
        if session_service is None:
            error_message = "Runner is missing session_service"
            raise ValueError(error_message)
        append_event = cast("Callable[..., Awaitable[Event]]", session_service.append_event)

        async def _append() -> None:
            for message in history:
                event = Event(author=self._event_author(message.role), content=self._to_content(message))
                await append_event(session=session, event=event)

        asyncio.run(_append())

    def _to_content(self, message: ChatMessage) -> types.Content:
        part = types.Part.from_text(text=message.content)
        return types.Content(role=message.role, parts=[part])

    def _event_author(self, role: str) -> str:
        if role == "assistant":
            return self._agent_name
        return "user"

    def _run(self, runner: Any, session_id: str, new_message: types.Content) -> tuple[list[Event], float]:
        start = perf_counter()
        events = list(self._run_function(runner, session_id, new_message, self._run_config))
        elapsed_seconds = perf_counter() - start
        return events, elapsed_seconds

    def _to_response(self, events: list[Event], elapsed_seconds: float) -> LLMResponse:
        content = self._extract_content(events)
        raw_events: list[dict[str, object]] = [self._dump_event(event) for event in events]
        raw_payload: dict[str, object] = {"events": raw_events, "config": self._run_metadata}
        tool_calls = normalize_tool_calls(raw_payload)
        return LLMResponse(
            content=content,
            model=self._model,
            raw=raw_payload,
            elapsed_seconds=elapsed_seconds,
            call_count=1,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _extract_content(events: list[Event]) -> str:
        for event in reversed(events):
            if event.content and event.content.parts:
                texts: list[str] = []
                for part in event.content.parts:
                    text = getattr(part, "text", None)
                    if text is not None:
                        texts.append(text)
                if texts:
                    return "\n".join(texts)
            if event.content:
                return str(event.content)
        return ""

    @staticmethod
    def _dump_event(event: Event) -> dict[str, object]:
        if hasattr(event, "model_dump"):
            return event.model_dump()
        return event.__dict__.copy()

    @staticmethod
    def _default_runner_factory(agent: Agent) -> object:
        return InMemoryRunner(agent=agent)

    def _default_run_function(
        self, runner: Any, session_id: str, new_message: types.Content, run_config: RunConfig,
    ) -> Iterable[Event]:
        run_callable = cast("Callable[..., Iterable[Event]]", runner.run)
        return run_callable(
            user_id=self._user_id, session_id=session_id, new_message=new_message, run_config=run_config,
        )
