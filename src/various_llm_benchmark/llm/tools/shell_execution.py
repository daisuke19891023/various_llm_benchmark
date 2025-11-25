"""Safe shell execution tool registration and helpers."""

from __future__ import annotations

import subprocess as sp
from dataclasses import dataclass

from various_llm_benchmark.llm.tools.registry import (
    SHELL_TAG,
    NativeToolType,
    ToolCategory,
    ToolRegistration,
    register_tool,
)

# Allowlisted commands mapped to permitted flag arguments.
ALLOWED_COMMANDS: dict[str, set[str]] = {
    "echo": set(),
    "ls": {"-a", "-l"},
    "sleep": set(),
}

TIMEOUT_EXIT_CODE = -1
MISSING_COMMAND_EXIT_CODE = 127


@dataclass
class ShellExecutionResult:
    """Captured outputs from a shell command."""

    stdout: str
    stderr: str
    exit_code: int


def _validate_arguments(command: str, args: list[str]) -> list[str]:
    """Validate command and arguments against the allowlist."""
    try:
        allowed_flags = ALLOWED_COMMANDS[command]
    except KeyError as exc:
        msg = f"Command '{command}' is not permitted."
        raise ValueError(msg) from exc

    validated_args: list[str] = []
    for arg in args:
        if arg.startswith("-") and arg not in allowed_flags:
            msg = f"Flag '{arg}' is not permitted for command '{command}'."
            raise ValueError(msg)
        validated_args.append(arg)
    return validated_args


def _ensure_text(value: str | bytes | None) -> str:
    """Return textual output regardless of the underlying subprocess type."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode()
    return value


def run_shell(
    command: str,
    args: list[str] | None = None,
    *,
    timeout_seconds: float = 5.0,
) -> dict[str, int | str]:
    """Execute an allowlisted shell command and capture outputs."""
    if timeout_seconds <= 0:
        msg = "timeout_seconds must be positive."
        raise ValueError(msg)

    provided_args = list(args or [])
    validated_args = _validate_arguments(command, provided_args)
    try:
        completed_process = sp.run(  # noqa: S603
            [command, *validated_args],
            capture_output=True,
            shell=False,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        result = ShellExecutionResult(
            stdout=_ensure_text(completed_process.stdout),
            stderr=_ensure_text(completed_process.stderr),
            exit_code=completed_process.returncode,
        )
    except sp.TimeoutExpired as exc:
        stdout = _ensure_text(exc.stdout)
        stderr = _ensure_text(exc.stderr)
        if stderr:
            stderr += "\n"
        stderr += f"Command timed out after {timeout_seconds} seconds."
        result = ShellExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=TIMEOUT_EXIT_CODE,
        )
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        result = ShellExecutionResult(
            stdout="",
            stderr=str(exc),
            exit_code=MISSING_COMMAND_EXIT_CODE,
        )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
    }


def _build_input_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": sorted(ALLOWED_COMMANDS.keys()),
                "description": "実行を許可するコマンド",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "コマンドに渡す引数の配列",
            },
            "timeout_seconds": {
                "type": "number",
                "minimum": 0.01,
                "description": "タイムアウト (秒)",
                "default": 5.0,
            },
        },
        "required": ["command"],
    }


def _build_provider_overrides() -> dict[str, object]:
    return {
        "openai": {"type": "bash"},
        "anthropic": {"type": "bash", "name": "shell-execute"},
    }


def _register_tool() -> None:
    """Register the shell execution tool with provider overrides."""
    description = "許可されたコマンドだけをタイムアウト付きで実行し、出力を返す"
    registration = ToolRegistration(
        id="shell/execute",
        name="shell-execute",
        description=description,
        input_schema=_build_input_schema(),
        tags={"execution", SHELL_TAG},
        native_type=NativeToolType.SHELL,
        provider_overrides=_build_provider_overrides(),
        handler=run_shell,
        category=ToolCategory.BUILTIN,
    )
    try:
        register_tool(registration)
    except ValueError:
        return


_register_tool()
