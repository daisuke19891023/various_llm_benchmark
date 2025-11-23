
from dataclasses import dataclass
from typing import Any

from google.adk.events.event import Event

@dataclass
class Session:
    app_name: str
    user_id: str
    id: str
    state: dict[str, Any]
    events: list[Event]
    last_update_time: float
