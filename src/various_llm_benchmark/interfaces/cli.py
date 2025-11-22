from __future__ import annotations

import typer

from various_llm_benchmark.interfaces.commands.agent import agent_app
from various_llm_benchmark.interfaces.commands.agent_sdk import agent_sdk_app
from various_llm_benchmark.interfaces.commands.claude import claude_app
from various_llm_benchmark.interfaces.commands.openai import openai_app

app = typer.Typer(help="各種LLMやエージェントをCLIから呼び出すためのツールです。")
app.add_typer(openai_app, name="openai")
app.add_typer(claude_app, name="claude")
app.add_typer(agent_app, name="agent")
app.add_typer(agent_sdk_app, name="agent-sdk")


def main() -> None:
    """Entrypoint for the CLI application."""
    app()


if __name__ == "__main__":
    main()
