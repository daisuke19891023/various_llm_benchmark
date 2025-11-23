
from google.genai.types import Content

class Event:
    author: str
    content: Content | None

    def __init__(self, *, author: str, content: Content | None = None) -> None: ...

    def model_dump(self) -> dict[str, object]: ...
