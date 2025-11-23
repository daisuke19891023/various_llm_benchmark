
from dataclasses import dataclass

@dataclass
class FileData:
    file_uri: str
    mime_type: str | None = None


class Part:
    text: str | None
    file_data: FileData | None

    def __init__(self, *, text: str | None = None, file_data: FileData | None = None) -> None: ...

    @classmethod
    def from_text(cls, *, text: str) -> Part: ...

    @classmethod
    def from_uri(cls, *, file_uri: str, mime_type: str | None = None) -> Part: ...


class Content:
    role: str
    parts: list[Part]

    def __init__(self, *, role: str, parts: list[Part]) -> None: ...


__all__ = ["Content", "FileData", "Part"]
