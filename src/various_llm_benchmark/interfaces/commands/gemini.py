from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import typer
from rich.console import Console
from google.genai import Client

from various_llm_benchmark.interfaces.commands.common import build_messages
from various_llm_benchmark.media.images import read_image_file
from various_llm_benchmark.media.audio_video import read_audio_or_video_file
from various_llm_benchmark.llm.providers.gemini.client import GeminiLLMClient
from various_llm_benchmark.prompts.prompt import PromptTemplate, load_provider_prompt
from various_llm_benchmark.settings import settings

gemini_app = typer.Typer(help="Geminiモデルを呼び出します。")
console = Console()

HISTORY_OPTION: list[str] | None = typer.Option(
    None,
    help="'role:content' 形式の履歴を複数回指定できます。",
    show_default=False,
)
IMAGE_ARGUMENT = typer.Argument(
    ...,
    exists=True,
    readable=True,
    dir_okay=False,
    help="解析する画像ファイルのパス",
)
MEDIA_ARGUMENT = typer.Argument(
    ...,
    exists=True,
    readable=True,
    dir_okay=False,
    help="解析する音声または動画ファイルのパス",
)


@lru_cache(maxsize=1)
def _prompt_template() -> PromptTemplate:
    return load_provider_prompt("llm", "gemini")


def _client() -> GeminiLLMClient:
    client = Client(api_key=settings.gemini_api_key.get_secret_value())
    return GeminiLLMClient(
        client,
        settings.gemini_model,
        temperature=settings.default_temperature,
        thinking_level=settings.gemini_thinking_level,
    )


@gemini_app.command("complete")
def gemini_complete(
    prompt: str,
    model: str | None = typer.Option(None, help="モデル名を上書きします。"),
    thinking_level: str | None = typer.Option(None, help="thinking level を指定します。"),
) -> None:
    """Generate a single-turn response with Gemini."""
    with console.status("Geminiで応答生成中...", spinner="dots"):
        response = _client().generate(
            _prompt_template().to_prompt_text(prompt),
            model=model,
            thinking_level=thinking_level,
        )
    console.print(response.content)


@gemini_app.command("chat")
def gemini_chat(
    prompt: str,
    history: list[str] | None = HISTORY_OPTION,
    model: str | None = typer.Option(None, help="モデル名を上書きします。"),
    thinking_level: str | None = typer.Option(None, help="thinking level を指定します。"),
) -> None:
    """Generate a chat response with optional history."""
    messages = build_messages(prompt, history or [])
    with console.status("Geminiで履歴付き応答を生成中...", spinner="dots"):
        response = _client().chat(
            messages,
            model=model,
            system_instruction=_prompt_template().system,
            thinking_level=thinking_level,
        )
    console.print(response.content)


@gemini_app.command("vision")
def gemini_vision(
    prompt: str,
    image_path: Path = IMAGE_ARGUMENT,
    model: str | None = typer.Option(None, help="モデル名を上書きします。"),
    thinking_level: str | None = typer.Option(None, help="thinking level を指定します。"),
) -> None:
    """Analyze an image with a Gemini model."""
    resolved_path = Path(image_path)
    image_input = read_image_file(resolved_path)
    with console.status("Geminiで画像を解析中...", spinner="dots"):
        response = _client().vision(
            prompt,
            image_input,
            model=model,
            system_prompt=_prompt_template().system,
            thinking_level=thinking_level,
    )
    console.print(response.content)


@gemini_app.command("multimodal")
def gemini_multimodal(
    prompt: str,
    media_paths: list[Path] = MEDIA_ARGUMENT,
    model: str | None = typer.Option(None, help="モデル名を上書きします。"),
    thinking_level: str | None = typer.Option(None, help="thinking level を指定します。"),
) -> None:
    """Analyze audio or video with a Gemini model."""
    media_inputs = [read_audio_or_video_file(path) for path in media_paths]
    with console.status("Geminiでメディアを解析中...", spinner="dots"):
        response = _client().multimodal(
            prompt,
            media_inputs,
            model=model,
            system_prompt=_prompt_template().system,
            thinking_level=thinking_level,
        )
    console.print(response.content)
