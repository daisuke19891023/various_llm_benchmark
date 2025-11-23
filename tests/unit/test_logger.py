from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import SecretStr

from various_llm_benchmark.logger import BaseComponent, configure_logging, get_logger
from various_llm_benchmark.settings import Settings


class SampleComponent(BaseComponent):
    """Lightweight component for exercising logging helpers."""


def _settings(tmp_path: Path, *, destination: Literal["stdout", "file", "both"] = "file") -> Settings:
    """Create settings tailored for logging tests."""
    return Settings(
        openai_api_key=SecretStr("test-key"),
        gemini_api_key=SecretStr("test-key"),
        log_destination=destination,
        log_file_path=str(tmp_path / "app.log"),
        log_level="INFO",
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
    """BaseComponent emits structured start events with metadata."""
    settings = _settings(tmp_path, destination="stdout")
    configure_logging(settings, force=True)
    component = SampleComponent()

    component.log_start("sample", note="example")
    stdout = capsys.readouterr().out.splitlines()[-1]
    payload = json.loads(stdout)
    assert payload["event"] == "start"
    assert payload["component"] == "SampleComponent"
    assert payload["action"] == "sample"
    assert payload["note"] == "example"
