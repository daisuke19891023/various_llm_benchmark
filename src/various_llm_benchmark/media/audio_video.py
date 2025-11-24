"""Helpers for loading audio and video files for Gemini inputs."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from various_llm_benchmark.models import MediaInput


def read_audio_or_video_file(media_path: Path) -> MediaInput:
    """Read an audio or video file and encode it to base64 inline data."""
    return _read_binary_media(media_path, allowed_prefixes=("audio/", "video/"))


def read_audio_file(media_path: Path) -> MediaInput:
    """Read an audio file and encode it to base64 inline data."""
    return _read_binary_media(media_path, allowed_prefixes=("audio/",))


def read_video_file(media_path: Path) -> MediaInput:
    """Read a video file and encode it to base64 inline data."""
    return _read_binary_media(media_path, allowed_prefixes=("video/",))


def _read_binary_media(media_path: Path, *, allowed_prefixes: tuple[str, ...]) -> MediaInput:
    resolved_path = Path(media_path)
    if not resolved_path.is_file():
        message = f"Media file not found: {resolved_path}"
        raise FileNotFoundError(message)

    media_type, _ = mimetypes.guess_type(resolved_path)
    if media_type is None or not media_type.startswith(allowed_prefixes):
        message = f"Unsupported media type for file: {resolved_path}"
        raise ValueError(message)

    data = resolved_path.read_bytes()
    encoded = base64.b64encode(data).decode()
    return MediaInput(media_type=media_type, data=encoded)

