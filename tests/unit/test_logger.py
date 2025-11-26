from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import SecretStr

from various_llm_benchmark.logger import BaseComponent, configure_logging, get_logger
from various_llm_benchmark.settings import Settings


class SampleComponent(BaseComponent):
    """Lightweight component for exercising logging helpers."""


def _settings(
    tmp_path: Path,
    *,
    destination: Literal["stdout", "file", "both"] = "file",
    allow_sensitive_logging: bool = False,
) -> Settings:
    """Create settings tailored for logging tests."""
    return Settings(
        openai_api_key=SecretStr("test-key"),
        gemini_api_key=SecretStr("test-key"),
        log_destination=destination,
        log_file_path=str(tmp_path / "app.log"),
        log_level="INFO",
        allow_sensitive_logging=allow_sensitive_logging,
    )


def test_configure_logging_writes_structured_file(tmp_path: Path, monkeypatch: Any) -> None:
    """configure_logging writes JSON lines to the configured log file."""
    log_file = tmp_path / "app.log"
    monkeypatch.setenv("LOG_DESTINATION", "file")
    monkeypatch.setenv("LOG_FILE_PATH", str(log_file))
    settings = _settings(tmp_path)
    configure_logging(settings, force=True)
    logger = get_logger(component="TestComponent")

    logger.info("hello", marker="value")
    handlers = [handler for handler in logging.getLogger().handlers if isinstance(handler, logging.FileHandler)]
    assert handlers, "File handler should be configured"
    handler = handlers[0]
    handler.flush()
    log_path = Path(handler.baseFilename)
    payload = json.loads(log_path.read_text().splitlines()[-1])
    logging.shutdown()

    assert payload["event"] == "hello"
    assert payload["marker"] == "value"
    assert payload["component"] == "TestComponent"


def test_base_component_logs_start_event(tmp_path: Path, capsys: Any) -> None:
    """BaseComponent emits readable console logs with metadata."""
    settings = _settings(tmp_path, destination="stdout")
    configure_logging(settings, force=True)
    component = SampleComponent()

    component.log_start("sample", note="example")
    stdout = capsys.readouterr().out
    plain_stdout = re.sub(r"\x1b\[[\d;]*m", "", stdout)

    assert "SampleComponent" in plain_stdout
    assert "start" in plain_stdout
    assert "action" in plain_stdout
    assert "sample" in plain_stdout
    assert "note" in plain_stdout
    assert "example" in plain_stdout
    assert "INFO" in plain_stdout
    assert "\x1b[32m" in stdout


def test_log_io_emphasizes_direction(tmp_path: Path, capsys: Any) -> None:
    """log_io should render direction markers for readability."""
    settings = _settings(tmp_path, destination="stdout")
    configure_logging(settings, force=True)
    component = SampleComponent()

    component.log_io(direction="input", prompt="hello world")
    stdout = capsys.readouterr().out
    plain_stdout = re.sub(r"\x1b\[[\d;]*m", "", stdout)

    assert "io" in plain_stdout
    assert "prompt" in plain_stdout
    assert "hello world" not in plain_stdout
    assert "<redacted text length=11>" in plain_stdout
    assert "⬅ input" in plain_stdout
    assert "\x1b[36m" in stdout


def test_log_io_allows_full_payload_when_enabled(tmp_path: Path, capsys: Any) -> None:
    """Sensitive payloads are only logged verbatim when explicitly enabled."""
    settings = _settings(tmp_path, destination="stdout", allow_sensitive_logging=True)
    configure_logging(settings, force=True)
    component = SampleComponent()

    visible_prompt = "example prompt text"
    component.log_io(direction="input", prompt=visible_prompt)
    stdout = capsys.readouterr().out
    plain_stdout = re.sub(r"\x1b\[[\d;]*m", "", stdout)

    assert "prompt" in plain_stdout
    assert visible_prompt in plain_stdout


def test_configure_logging_preserves_active_settings(tmp_path: Path, capsys: Any) -> None:
    """Subsequent configure_logging calls should not revert active settings."""
    sensitive_settings = _settings(tmp_path, destination="stdout", allow_sensitive_logging=True)
    configure_logging(sensitive_settings, force=True)

    configure_logging()
    component = SampleComponent()

    visible_prompt = "persisted prompt"
    component.log_io(direction="input", prompt=visible_prompt)
    stdout = capsys.readouterr().out
    plain_stdout = re.sub(r"\x1b\[[\d;]*m", "", stdout)

    assert "prompt" in plain_stdout
    assert visible_prompt in plain_stdout
    assert "<redacted" not in plain_stdout
