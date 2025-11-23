from __future__ import annotations

import logging
import sys
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import structlog
from structlog.processors import CallsiteParameter
from structlog.stdlib import BoundLogger

from various_llm_benchmark.settings import Settings, settings

if TYPE_CHECKING:
    from structlog.typing import Processor

BindableLogger = BoundLogger
Processor = structlog.types.Processor

_LEVELS: dict[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

_CONFIG_STATE: dict[str, bool] = {"configured": False}


def _log_level(level_name: str) -> int:
    return _LEVELS.get(level_name.upper(), logging.INFO)


def _build_handlers(app_settings: Settings) -> list[logging.Handler]:
    handlers: list[logging.Handler] = []
    if app_settings.log_destination in {"stdout", "both"}:
        handlers.append(logging.StreamHandler(sys.stdout))

    if app_settings.log_destination in {"file", "both"}:
        log_path = Path(app_settings.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    return handlers


def configure_logging(app_settings: Settings | None = None, *, force: bool = False) -> None:
    """Configure structlog and stdlib logging outputs based on :class:`Settings`."""
    if force:
        structlog.reset_defaults()
        _CONFIG_STATE["configured"] = False

    if _CONFIG_STATE["configured"] and not force:
        return

    active_settings = app_settings or settings
    handlers = _build_handlers(active_settings)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(handler)
    root_logger.setLevel(_log_level(active_settings.log_level))

    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if active_settings.log_verbose:
        processors.append(
            structlog.processors.CallsiteParameterAdder(  # type: ignore[arg-type]
                parameters=(CallsiteParameter.FUNC_NAME, CallsiteParameter.LINENO),
            ),
        )

    processors.extend(
        [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ],
    )

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    _CONFIG_STATE["configured"] = True


def get_logger(*, component: str | None = None) -> BindableLogger:
    """Return a structlog logger bound to the optional component name."""
    configure_logging()
    logger = structlog.get_logger()
    if component:
        return logger.bind(component=component)
    return logger


class BaseComponent:
    """Mixin providing a pre-configured structlog logger and helpers."""

    @cached_property
    def logger(self) -> BindableLogger:
        """Return a logger bound with the current class name."""
        return get_logger(component=self.__class__.__name__)

    def log_start(self, action: str, **kwargs: object) -> None:
        """Emit a standardized start event."""
        self.logger.info("start", action=action, **kwargs)

    def log_end(self, action: str, **kwargs: object) -> None:
        """Emit a standardized completion event."""
        self.logger.info("end", action=action, **kwargs)

    def log_io(self, direction: Literal["input", "output"], **kwargs: object) -> None:
        """Emit structured input/output payloads."""
        self.logger.info("io", direction=direction, **kwargs)
