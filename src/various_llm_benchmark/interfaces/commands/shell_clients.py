"""Resolver for executing the shell tool per provider."""
from __future__ import annotations

import importlib
from typing import Literal, Protocol, TypedDict, cast

from various_llm_benchmark.llm.tools.registry import SHELL_TAG, ToolRegistration
from various_llm_benchmark.llm.tools.selector import ToolSelector

ProviderName = Literal["openai", "anthropic"]
ShellProviderName = ProviderName


class ShellResult(TypedDict):
    """Structured output from the shell tool."""

    stdout: str
    stderr: str
    exit_code: int


class ShellExecutor(Protocol):
    """Callable protocol for executing shell commands."""

    def __call__(self, command: str, args: list[str] | None, timeout_seconds: float) -> ShellResult:
        """Execute the shell command and return structured results."""
        ...


def resolve_shell_client(provider: ProviderName) -> ShellExecutor:
    """Return an executor for the shell tool scoped to the provider."""
    _ensure_shell_tool_registered()
    selector = ToolSelector()
    shell_tool = selector.select_one(tags=[SHELL_TAG])
    _validate_provider_support(shell_tool, provider)

    def _executor(command: str, args: list[str] | None, timeout_seconds: float) -> ShellResult:
        result = shell_tool.handler(command=command, args=args or [], timeout_seconds=timeout_seconds)
        return cast("ShellResult", result)

    return _executor


def _validate_provider_support(registration: ToolRegistration, provider: ProviderName) -> None:
    override = registration.provider_overrides.get(provider)
    if not isinstance(override, dict):
        msg = f"Shell tool is not configured for provider '{provider}'."
        raise TypeError(msg)


def _ensure_shell_tool_registered() -> None:
    selector = ToolSelector()
    if selector.select(tags=[SHELL_TAG]):
        return
    importlib.import_module("various_llm_benchmark.llm.tools.shell_execution")
