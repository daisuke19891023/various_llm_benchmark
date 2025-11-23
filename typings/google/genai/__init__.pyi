
from typing import Any, Protocol

from . import types

class _ModelsClient(Protocol):
    def generate_content(self, *args: Any, **kwargs: Any) -> Any: ...
    def embed_content(self, *args: Any, **kwargs: Any) -> Any: ...

class Client:
    models: _ModelsClient

    def __init__(self, *, api_key: str | None = None) -> None: ...


def configure(*, api_key: str | None = None) -> None: ...

__all__ = ["Client", "configure", "types"]
