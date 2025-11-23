from __future__ import annotations

from typing import TYPE_CHECKING, Any

import various_llm_benchmark.agents.providers.google_adk as google_adk_module

from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types

from various_llm_benchmark.agents.providers.google_adk import GoogleADKProvider
from various_llm_benchmark.models import ChatMessage, ImageInput

if TYPE_CHECKING:
    from collections.abc import Iterable


class RecordingSessionService:
    """In-memory session service substitute for tests."""

    def __init__(self) -> None:
        """Initialize in-memory containers."""
        self.created: list[tuple[str, str, str]] = []
        self.appended: list[Event] = []

    def create_session_sync(self, *, app_name: str, user_id: str, session_id: str) -> Session:
        """Create a session and record the parameters used."""
        self.created.append((app_name, user_id, session_id))
        return Session(app_name=app_name, user_id=user_id, id=session_id, state={}, events=[], last_update_time=0.0)

    async def append_event(self, session: Session, event: Event) -> Event:
        """Append an event to a session while tracking calls."""
        session.events.append(event)
        self.appended.append(event)
        return event


type RecordingRun = dict[str, Any]


class StubRunner:
    """Minimal runner that records invocations and yields canned events."""

    def __init__(self, *, responses: Iterable[Event]) -> None:
        """Store canned responses and initialize tracking lists."""
        self.session_service = RecordingSessionService()
        self.app_name = "test-app"
        self.calls: list[RecordingRun] = []
        self._responses = list(responses)

    def run(
        self,
        *,
        user_id: str,
        session_id: str,
        new_message: types.Content,
        run_config: Any,
    ) -> Iterable[Event]:
        """Record the call arguments and return canned events."""
        self.calls.append(
            {
                "user_id": user_id,
                "session_id": session_id,
                "new_message": new_message,
                "run_config": run_config,
            },
        )
        return list(self._responses)


def _text_event(text: str) -> Event:
    return Event(
        author="google_adk_agent",
        content=types.Content(role="model", parts=[types.Part.from_text(text=text)]),
    )


def test_complete_runs_runner_and_builds_agent(monkeypatch: Any) -> None:
    """Complete should build an agent and forward prompt to the runner."""
    created_agents: list[Any] = []
    times = iter([1.5, 2.5])
    monkeypatch.setattr(google_adk_module, "perf_counter", lambda: next(times))
    runner = StubRunner(responses=[_text_event("done")])

    def runner_factory(agent: Any) -> StubRunner:
        created_agents.append(agent)
        return runner

    provider = GoogleADKProvider(
        api_key="key",
        model="gemini-test",
        instructions="follow guidance",
        runner_factory=runner_factory,
    )

    response = provider.complete("hello")

    assert response.content == "done"
    assert response.model == "gemini-test"
    assert response.elapsed_seconds == 1.0
    assert response.call_count == 1
    assert response.tool_calls == []
    assert created_agents[0].instruction == "follow guidance"
    assert runner.calls
    assert runner.calls[0]["new_message"].parts[0].text == "hello"


def test_chat_seeds_history_and_uses_last_message() -> None:
    """History should be appended to the session before sending the latest message."""
    runner = StubRunner(responses=[_text_event("reply")])
    provider = GoogleADKProvider(
        api_key="key",
        model="gemini-test",
        instructions="assist",
        runner_factory=lambda _agent: runner,
    )
    messages = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="assistant", content="previous"),
        ChatMessage(role="user", content="question"),
    ]

    response = provider.chat(messages)

    assert response.content == "reply"
    assert runner.session_service.appended
    first_event = runner.session_service.appended[0]
    assert first_event.content is not None
    assert first_event.content.parts
    assert first_event.content.parts[0].text == "sys"
    assert runner.session_service.appended[1].author == "google_adk_agent"
    assert runner.calls
    assert runner.calls[0]["new_message"].parts[0].text == "question"


def test_vision_builds_image_message() -> None:
    """Vision should include both text and image parts."""
    runner = StubRunner(responses=[_text_event("saw image")])
    provider = GoogleADKProvider(
        api_key="key",
        model="gemini-test",
        instructions="describe",
        runner_factory=lambda _agent: runner,
    )
    image = ImageInput(media_type="image/png", data="ZmFrZQ==")

    response = provider.vision("caption", image)

    assert response.content == "saw image"
    assert runner.calls
    parts = runner.calls[0]["new_message"].parts
    assert parts[0].text == "caption"
    assert parts[1].file_data
    assert parts[1].file_data.file_uri.startswith("data:image/png;base64,ZmFrZQ==")
