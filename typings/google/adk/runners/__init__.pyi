
from collections.abc import Iterable
from typing import Any

from google.adk.agents.agent import Agent
from google.adk.events.event import Event
from google.adk.agents.run_config import RunConfig

class InMemoryRunner:
    app_name: str
    session_service: Any

    def __init__(self, *, agent: Agent) -> None: ...

    def run(
        self,
        *,
        user_id: str,
        session_id: str,
        new_message: Any,
        run_config: RunConfig,
    ) -> Iterable[Event]: ...
