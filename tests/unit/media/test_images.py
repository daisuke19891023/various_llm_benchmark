from __future__ import annotations

import base64
from pathlib import Path

import pytest

from various_llm_benchmark.media.images import read_image_file


def test_read_image_file_encodes_payload(tmp_path: Path) -> None:
    """画像がbase64に変換されデータURLが生成されることを確認する."""
    image_path = Path(tmp_path / "sample.png")
    image_bytes = b"\x89PNG\r\n\x1a\npayload"
    image_path.write_bytes(image_bytes)

    image_input = read_image_file(image_path)

    assert image_input.media_type == "image/png"
    assert image_input.data == base64.b64encode(image_bytes).decode()
    assert image_input.as_data_url().startswith("data:image/png;base64,")


def test_read_image_file_rejects_non_image(tmp_path: Path) -> None:
    """画像以外のパスでは例外が送出されることを確認する."""
    text_path = Path(tmp_path / "not_image.txt")
    text_path.write_text("hello")

    with pytest.raises(ValueError, match="Unsupported image type"):
        read_image_file(text_path)
