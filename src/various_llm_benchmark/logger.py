from __future__ import annotations

import logging
import sys
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from rich.console import Console
from rich.logging import RichHandler
import structlog
from structlog.processors import CallsiteParameter
from structlog.stdlib import BoundLogger

from various_llm_benchmark.settings import Settings, settings

if TYPE_CHECKING:
    from structlog.typing import EventDict, Processor

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

_LEVEL_STYLES: dict[str, str] = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bright_red",
}


def _log_level(level_name: str) -> int:
    return _LEVELS.get(level_name.upper(), logging.INFO)


def _console_renderer(_logger: logging.Logger, _name: str, event_dict: EventDict) -> str:
    timestamp = event_dict.get("timestamp")
    level = str(event_dict.get("level", "")).upper()
    component = event_dict.get("component")
    event = event_dict.get("event")
    direction = event_dict.get("direction")

    details = {
        k: v
        for k, v in event_dict.items()
        if k not in {"timestamp", "level", "component", "event", "direction"}
    }

    level_style = _LEVEL_STYLES.get(level, "white")
    level_markup = f"[{level_style}]{level:>8}[/]" if level else ""
    direction_markup = ""
    if event == "io" and isinstance(direction, str):
        if direction == "input":
            direction_markup = "[cyan]⬅ input[/]"
        elif direction == "output":
            direction_markup = "[magenta]➡ output[/]"

    parts = [
        f"[dim]{timestamp}[/]" if timestamp else "",
        level_markup,
        f"[bold]{component}[/]" if component else "",
        f"[italic]{event}[/]" if event else "",
        direction_markup,
    ]

    if details:
        formatted_details = " ".join(f"[blue]{key}[/]=[white]{value}[/]" for key, value in sorted(details.items()))
        parts.append(formatted_details)

    rendered = " ".join(part for part in parts if part)
    return rendered or str(event)


def _shared_pre_chain(app_settings: Settings) -> list[Processor]:
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if app_settings.log_verbose:
        processors.append(
            structlog.processors.CallsiteParameterAdder(  # type: ignore[arg-type]
                parameters=(CallsiteParameter.FUNC_NAME, CallsiteParameter.LINENO),
            ),
        )

    return processors


def _build_handlers(app_settings: Settings) -> list[logging.Handler]:
    handlers: list[logging.Handler] = []

    pre_chain = _shared_pre_chain(app_settings)

    if app_settings.log_destination in {"stdout", "both"}:
        console_handler = RichHandler(
            console=Console(file=sys.stdout, force_terminal=True),
            rich_tracebacks=True,
            show_time=False,
            markup=True,
        )
        console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(  # type: ignore[arg-type]
                processor=_console_renderer,
                foreign_pre_chain=pre_chain,
            ),
        )
        handlers.append(console_handler)

    if app_settings.log_destination in {"file", "both"}:
        log_path = Path(app_settings.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(  # type: ignore[arg-type]
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=pre_chain,
            ),
        )
        handlers.append(file_handler)

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
        root_logger.addHandler(handler)
    root_logger.setLevel(_log_level(active_settings.log_level))

    processors: list[Processor] = _shared_pre_chain(active_settings)
    processors.extend(
        [
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
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
