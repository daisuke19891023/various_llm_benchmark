"""Shared helpers for loading and validating image inputs."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from various_llm_benchmark.models import ImageInput


def read_image_file(image_path: Path) -> ImageInput:
    """Read an image file and encode it into a base64 payload."""
    resolved_path = Path(image_path)
    if not resolved_path.is_file():
        message = f"Image file not found: {resolved_path}"
        raise FileNotFoundError(message)

    media_type, _ = mimetypes.guess_type(resolved_path)
    if media_type is None or not media_type.startswith("image/"):
        message = f"Unsupported image type for file: {resolved_path}"
        raise ValueError(message)

    data = resolved_path.read_bytes()
    encoded = base64.b64encode(data).decode()
    return ImageInput(media_type=media_type, data=encoded)

