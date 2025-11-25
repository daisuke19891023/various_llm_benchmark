from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from various_llm_benchmark.interfaces.commands.shell_clients import resolve_shell_client
from various_llm_benchmark.llm.tools.registry import SHELL_TAG, ToolRegistration
from various_llm_benchmark.llm.tools.selector import ToolSelector

if TYPE_CHECKING:
    from collections.abc import Iterator


def _shell_tool() -> ToolRegistration:
    selector = ToolSelector()
    return selector.select_one(tags=[SHELL_TAG])


@pytest.fixture
def restore_shell_tool_handler() -> Iterator[None]:
    """Restore the shell tool handler after each test."""
    tool = _shell_tool()
    original_handler = tool.handler
    yield
    tool.handler = original_handler


def test_resolve_shell_client_executes_command() -> None:
    """Shellツール経由でechoが実行できることを確認."""
    executor = resolve_shell_client("openai")

    result = executor("echo", args=["hello"], timeout_seconds=3.0)

    assert result["stdout"].strip() == "hello"
    assert result["exit_code"] == 0


def test_resolve_shell_client_forwards_arguments(restore_shell_tool_handler: None) -> None:
    """ハンドラーに引数が正しく渡ることを確認."""
    _ = restore_shell_tool_handler
    tool = _shell_tool()
    captured: dict[str, Any] = {}

    def fake_handler(command: str, args: list[str], timeout_seconds: float) -> dict[str, object]:
        captured.update({"command": command, "args": args, "timeout": timeout_seconds})
        return {"stdout": "ok", "stderr": "", "exit_code": 0}

    tool.handler = fake_handler

    executor = resolve_shell_client("anthropic")
    result = executor("ls", args=["-a"], timeout_seconds=4.5)

    assert result == {"stdout": "ok", "stderr": "", "exit_code": 0}
    assert captured == {"command": "ls", "args": ["-a"], "timeout": 4.5}


def test_resolve_shell_client_requires_provider_override(restore_shell_tool_handler: None) -> None:
    """プロバイダー設定が欠けている場合に例外が送出されることを確認."""
    _ = restore_shell_tool_handler
    tool = _shell_tool()
    original_overrides = tool.provider_overrides
    tool.provider_overrides = {"openai": original_overrides["openai"]}

    with pytest.raises(TypeError, match="Shell tool is not configured"):
        resolve_shell_client("anthropic")

    tool.provider_overrides = original_overrides
