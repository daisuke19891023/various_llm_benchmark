"""Unit tests for shell execution helper and tool registration."""

from __future__ import annotations

import pytest

from various_llm_benchmark.llm.tools.payloads import (
    to_anthropic_tools_payload,
    to_openai_tools_payload,
)
from various_llm_benchmark.llm.tools.selector import ToolSelector
from various_llm_benchmark.llm.tools.shell_execution import run_shell


def test_shell_execution_captures_stdout() -> None:
    """Echo command should return stdout and zero exit code."""
    result = run_shell("echo", ["hello"])

    stdout = result["stdout"]
    stderr = result["stderr"]
    exit_code = result["exit_code"]

    assert isinstance(stdout, str)
    assert isinstance(stderr, str)
    assert isinstance(exit_code, int)

    assert exit_code == 0
    assert stdout.strip() == "hello"
    assert stderr == ""


def test_shell_execution_reports_stderr_and_exit_code() -> None:
    """Failures should keep stderr content and non-zero exit code."""
    result = run_shell("ls", ["nonexistent-file-"])

    stderr = result["stderr"]
    exit_code = result["exit_code"]

    assert isinstance(stderr, str)
    assert isinstance(exit_code, int)

    assert exit_code != 0
    assert "nonexistent-file-" in stderr


def test_shell_execution_blocks_disallowed_flags() -> None:
    """Flags outside the allowlist should raise an error."""
    with pytest.raises(ValueError, match="Flag '-z' is not permitted"):
        run_shell("ls", ["-z"])


def test_shell_execution_times_out() -> None:
    """Long-running commands should be interrupted with sentinel exit code."""
    result = run_shell("sleep", ["1"], timeout_seconds=0.01)

    stderr = result["stderr"]
    exit_code = result["exit_code"]

    assert isinstance(stderr, str)
    assert isinstance(exit_code, int)

    assert exit_code == -1
    assert "timed out" in stderr


def test_shell_tool_registered_with_provider_overrides() -> None:
    """Shell tool should be registered and expose bash-native payloads."""
    selector = ToolSelector()
    shell_tool = selector.select_one(tags=["shell"], category=None)

    openai_payload = to_openai_tools_payload([shell_tool])
    anthropic_payload = to_anthropic_tools_payload([shell_tool])

    assert openai_payload == [{"type": "bash"}]
    assert anthropic_payload == [{"type": "bash", "name": "shell-execute"}]

