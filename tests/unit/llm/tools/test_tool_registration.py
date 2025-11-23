"""Ensure builtin tools are available without manual imports."""

from __future__ import annotations

import importlib
import sys

from various_llm_benchmark.llm import tools
from various_llm_benchmark.llm.tools import registry


def test_builtin_tools_register_on_package_import() -> None:
    """ToolSelector should see builtin entries after importing the package."""
    for module in (
        "various_llm_benchmark.llm.tools.code_execution",
        "various_llm_benchmark.llm.tools.shell_execution",
        "various_llm_benchmark.llm.tools.builtin_memory",
    ):
        sys.modules.pop(module, None)

    importlib.reload(registry)
    importlib.reload(tools)

    selector = tools.ToolSelector()

    shell_tool = selector.select_one(ids=["shell/execute"])
    assert shell_tool.name == "shell-execute"
