from __future__ import annotations

import typer

from various_llm_benchmark.interfaces.commands.agent import agent_app
from various_llm_benchmark.interfaces.commands.agent_sdk import agent_sdk_app
from various_llm_benchmark.interfaces.commands.claude import claude_app
from various_llm_benchmark.interfaces.commands.compare import compare_app
from various_llm_benchmark.interfaces.commands.dspy import dspy_app
from various_llm_benchmark.interfaces.commands.google_adk import adk_app
from various_llm_benchmark.interfaces.commands.gemini import gemini_app
from various_llm_benchmark.interfaces.commands.openai import openai_app
from various_llm_benchmark.interfaces.commands.pydantic_ai import pydantic_ai_app
from various_llm_benchmark.interfaces.commands.tools import tools_app

app = typer.Typer(help="各種LLMやエージェントをCLIから呼び出すためのツールです。")
app.add_typer(openai_app, name="openai")
app.add_typer(claude_app, name="claude")
app.add_typer(gemini_app, name="gemini")
app.add_typer(adk_app, name="google-adk")
app.add_typer(agent_app, name="agent")
app.add_typer(agent_sdk_app, name="agent-sdk")
app.add_typer(pydantic_ai_app, name="pydantic-ai")
app.add_typer(compare_app, name="compare")
app.add_typer(tools_app, name="tools")
app.add_typer(dspy_app, name="dspy")


def main() -> None:
    """Entrypoint for the CLI application."""
    app()


if __name__ == "__main__":
    main()
