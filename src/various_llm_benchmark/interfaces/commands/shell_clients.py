from __future__ import annotations

import importlib

from typing import Literal, Protocol, cast

from various_llm_benchmark.llm.tools import ToolSelector
from various_llm_benchmark.llm.tools.registry import SHELL_TAG, ToolCategory

ProviderName = Literal["openai", "anthropic"]


class ShellExecutor(Protocol):
    """Callable that executes an allowlisted shell command."""

    def __call__(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        timeout_seconds: float = 5.0,
    ) -> dict[str, int | str]:
        """Execute the shell command and return captured outputs."""
        ...


def _ensure_shell_tool_registered() -> None:
    """Import the shell execution module to register the tool."""
    importlib.import_module("various_llm_benchmark.llm.tools.shell_execution")


def resolve_shell_client(
    provider: ProviderName, *, category: ToolCategory = ToolCategory.BUILTIN,
) -> ShellExecutor:
    """Return a shell executor registered for the given provider.

    The executor is shared between providers because the local execution
    behaviour is identical, but validating the provider keeps the CLI
    signature explicit.
    """
    if provider not in ("openai", "anthropic"):
        msg = f"Unsupported shell provider: {provider}"
        raise ValueError(msg)

    _ensure_shell_tool_registered()

    selector = ToolSelector()
    registration = selector.select_one(
        category=category,
        names=["shell-execute"],
        tags=[SHELL_TAG],
        ids=["shell/execute"],
    )
    return cast("ShellExecutor", registration.handler)


__all__ = ["ProviderName", "ShellExecutor", "resolve_shell_client"]
