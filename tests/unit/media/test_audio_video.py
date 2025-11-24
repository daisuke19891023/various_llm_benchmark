"""Tests for audio and video media helpers."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

import pytest

from various_llm_benchmark.media.audio_video import (
    read_audio_file,
    read_audio_or_video_file,
    read_video_file,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_read_audio_file_encodes_bytes(tmp_path: Path) -> None:
    """Audio files should be accepted and encoded as base64."""
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"abc")

    media = read_audio_file(audio_file)

    assert media.media_type.startswith("audio/")
    assert base64.b64decode(media.data) == b"abc"


def test_read_video_file_encodes_bytes(tmp_path: Path) -> None:
    """Video files should be accepted and encoded as base64."""
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(b"xyz")

    media = read_video_file(video_file)

    assert media.media_type.startswith("video/")
    assert base64.b64decode(media.data) == b"xyz"


def test_read_audio_or_video_rejects_other_types(tmp_path: Path) -> None:
    """Non-audio/video files should raise an error."""
    document = tmp_path / "note.txt"
    document.write_text("hello")

    with pytest.raises(ValueError, match="Unsupported media type"):
        read_audio_or_video_file(document)

