"""Safe code execution tool registration."""

from collections.abc import Iterable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import typing as t
from typing import Any, TypeGuard
from io import StringIO
import runpy
import tempfile
from pathlib import Path

from google.genai import types as genai_types

from various_llm_benchmark.llm.tools.registry import (
    CODE_EXECUTION_TAG,
    NativeToolType,
    ToolCategory,
    ToolRegistration,
    get_tool,
    register_tool,
)

ALLOWED_LANGUAGES: set[str] = {"python"}


class ToolExecutionContext:
    """Allow controlled access to registered tools inside executed code."""

    def __init__(self, allowed_tool_ids: list[str]) -> None:
        """Cache the permitted tools accessible to executed snippets."""
        self._tools: dict[str, ToolRegistration] = {}
        for tool_id in allowed_tool_ids:
            tool = get_tool(tool_id)
            self._tools[tool.id] = tool
            self._tools[tool.name] = tool

    def call_tool(self, identifier: str, /, **kwargs: Any) -> Any:
        """Invoke a permitted tool handler by id or name."""
        try:
            tool = self._tools[identifier]
        except KeyError as exc:
            msg = f"Tool '{identifier}' is not allowed in this execution context."
            raise ValueError(msg) from exc
        return tool.handler(**kwargs)

    @staticmethod
    def coerce_list(value: Any) -> list[Any]:
        """Convert arbitrary values to a list for simpler aggregation."""
        if value is None:
            return []
        if isinstance(value, list):
            return t.cast("list[t.Any]", value)
        if isinstance(value, (str, bytes)):
            return [value]
        if _is_iterable_any(value):
            result: list[Any] = list(value)
            return result
        return [value]

    @staticmethod
    def limit_items(items: list[Any] | tuple[Any, ...], limit: int) -> list[Any]:
        """Return at most ``limit`` items while preserving order."""
        if limit < 0:
            msg = "limit must be non-negative"
            raise ValueError(msg)
        return list(items)[:limit]


def _is_iterable_any(value: object) -> TypeGuard[Iterable[Any]]:
    """Return True when the value is an iterable of any items."""
    return isinstance(value, Iterable)


@dataclass
class ExecutionResult:
    """Captured outputs from a code execution."""

    stdout: str
    stderr: str
    exit_code: int


class SafeCodeExecutor:
    """Restricted code execution using in-process sandboxing."""

    def __init__(self, *, timeout_seconds: float = 5.0) -> None:
        """Configure executor defaults such as timeouts."""
        self._timeout = timeout_seconds

    def run(
        self,
        language: str,
        code: str,
        *,
        timeout_seconds: float | None = None,
        allowed_tool_ids: list[str] | None = None,
    ) -> ExecutionResult:
        """Execute code for a whitelisted language and capture outputs."""
        if language not in ALLOWED_LANGUAGES:
            msg = f"Unsupported language: {language}"
            raise ValueError(msg)

        _ = timeout_seconds or self._timeout
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        context = ToolExecutionContext(allowed_tool_ids) if allowed_tool_ids else None
        init_globals: dict[str, object] = {}
        if context:
            init_globals = {
                "call_tool": context.call_tool,
                "coerce_list": context.coerce_list,
                "limit_items": context.limit_items,
            }

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                script_path = Path(temp_dir) / "snippet.py"
                script_path.write_text(code)
                with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                    runpy.run_path(str(script_path), run_name="__main__", init_globals=init_globals)
            exit_code = 0
        except Exception as exc:
            stderr_buffer.write(str(exc))
            exit_code = 1

        return ExecutionResult(
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue(),
            exit_code=exit_code,
        )


_executor = SafeCodeExecutor()


def run_code(
    language: str,
    code: str,
    timeout_seconds: float = 5.0,
    tool_ids: list[str] | None = None,
) -> dict[str, object]:
    """Execute code using the shared executor and return captured outputs."""
    result = _executor.run(
        language,
        code,
        timeout_seconds=timeout_seconds,
        allowed_tool_ids=tool_ids,
    )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
    }


def _build_provider_overrides() -> dict[str, object]:
    """Construct provider-specific function declarations for registration."""
    return {
        "openai": {"type": "code_interpreter"},
        "anthropic": {"type": "code_execution_20250825", "name": "code_execution"},
        "gemini": genai_types.Tool(code_execution=genai_types.ToolCodeExecution()),
    }


def _register_tool() -> None:
    """Register the safe code execution tool with provider overrides."""
    schema: dict[str, object] = t.cast(
        "dict[str, object]",
        {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "enum": sorted(ALLOWED_LANGUAGES),
                    "description": "実行を許可する言語 (python)",
                },
                "code": {"type": "string", "description": "実行するコード"},
                "timeout_seconds": {
                    "type": "number",
                    "description": "タイムアウト (秒)",
                    "minimum": 1,
                    "default": 5.0,
                },
                "tool_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "コード内で呼び出しを許可するツールIDのリスト",
                },
            },
            "required": ["language", "code"],
        },
    )
    description = (
        "制限されたサンドボックスでPythonコードを実行し、必要に応じて他のツール呼び出しを組み合わせる"
    )
    registration = ToolRegistration(
        id="code/execute",
        name="code-execute",
        description=description,
        input_schema=schema,
        native_type=NativeToolType.CODE_EXECUTION,
        provider_overrides=_build_provider_overrides(),
        handler=run_code,
        category=ToolCategory.BUILTIN,
        tags={"code", "execution", CODE_EXECUTION_TAG},
    )
    try:
        register_tool(registration)
    except ValueError:
        return


_register_tool()

