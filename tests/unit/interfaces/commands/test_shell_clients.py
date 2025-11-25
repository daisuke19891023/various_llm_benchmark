from typing import cast

from various_llm_benchmark.interfaces.commands.shell_clients import resolve_shell_client


def test_resolve_shell_client_executes_allowlisted_command() -> None:
    """登録済みのシェルツールが呼び出せることを確認する."""
    executor = resolve_shell_client("openai")

    result = executor("echo", ["hello"], timeout_seconds=1)

    assert cast("int", result["exit_code"]) == 0
    assert cast("str", result["stderr"]) == ""
    assert cast("str", result["stdout"]).strip() == "hello"


def test_resolve_shell_client_accepts_anthropic_provider() -> None:
    """Anthropic指定でも同じ実行器を取得できる."""
    executor = resolve_shell_client("anthropic")

    result = executor("echo", ["hi"], timeout_seconds=1)

    assert cast("int", result["exit_code"]) == 0
    assert cast("str", result["stdout"]).strip() == "hi"
