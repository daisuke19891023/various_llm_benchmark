from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from various_llm_benchmark.llm.providers.anthropic.client import AnthropicLLMClient
from various_llm_benchmark.llm.providers.gemini.client import GeminiLLMClient
from various_llm_benchmark.llm.providers.openai.client import OpenAILLMClient
from various_llm_benchmark.models import ImageInput, MediaInput


class StubOpenAIResponse:
    """Stub response object for OpenAI client tests."""

    def __init__(self) -> None:
        """Initialize stub response fields."""
        self.output = {"text": {"value": "openai-image"}}
        self.model = "gpt-4o"

    def model_dump(self) -> dict[str, object]:
        """Return a minimal dictionary representation."""
        return {"raw": True}


class StubOpenAIResponses:
    """Stub collection for response creation calls."""

    def __init__(self) -> None:
        """Prepare a container for captured keyword arguments."""
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> StubOpenAIResponse:
        """Capture kwargs and return a stub response."""
        self.kwargs = kwargs
        return StubOpenAIResponse()


class StubOpenAI:
    """Stub OpenAI client exposing the responses API."""

    def __init__(self) -> None:
        """Expose a stub responses resource."""
        self.responses = StubOpenAIResponses()


def test_openai_vision_builds_data_url() -> None:
    """OpenAI vision calls include text and image payloads."""
    stub_client = StubOpenAI()
    client = OpenAILLMClient(cast("Any", stub_client), "gpt-default", temperature=0.1)
    image = ImageInput(media_type="image/png", data="YWJj")

    response = client.vision("describe", image, system_prompt="system")

    assert response.content == "openai-image"
    assert response.model == "gpt-4o"
    assert stub_client.responses.kwargs is not None
    content = stub_client.responses.kwargs["input"][0]["content"]  # type: ignore[index]
    assert content[0]["text"] == "system\n\ndescribe"  # type: ignore[index]
    assert content[1]["image_url"].endswith(image.data)  # type: ignore[index]


class StubAnthropicResponse:
    """Stub response payload returned by Anthropic client."""

    def __init__(self) -> None:
        """Initialize stub content and model name."""
        self.content = [{"text": "anthropic-image"}]
        self.model = "claude-3"

    def model_dump(self) -> dict[str, object]:
        """Return stub raw payload."""
        return {"raw": True}


class StubAnthropicMessages:
    """Stub messages resource that records create calls."""

    def __init__(self) -> None:
        """Prepare storage for recorded calls."""
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> StubAnthropicResponse:
        """Capture message creation arguments."""
        self.kwargs = kwargs
        return StubAnthropicResponse()


class StubAnthropic:
    """Stub Anthropic client exposing messages."""

    def __init__(self) -> None:
        """Attach stub message resource."""
        self.messages = StubAnthropicMessages()


def test_anthropic_vision_uses_base64_source() -> None:
    """Anthropic vision calls embed base64 image sources."""
    stub_client = StubAnthropic()
    client = AnthropicLLMClient(cast("Any", stub_client), "claude-default", temperature=0.3)
    image = ImageInput(media_type="image/jpeg", data="ZGF0YQ==")

    response = client.vision("draw", image, model="claude-3-opus", system_prompt="sys")

    assert response.content == "anthropic-image"
    assert response.model == "claude-3"
    assert stub_client.messages.kwargs is not None
    message_content = stub_client.messages.kwargs["messages"][0]["content"]  # type: ignore[index]
    assert message_content[0]["text"] == "draw"  # type: ignore[index]
    source = message_content[1]["source"]  # type: ignore[index]
    assert source["media_type"] == "image/jpeg"
    assert source["data"] == image.data
    assert stub_client.messages.kwargs["system"] == "sys"  # type: ignore[index]


@dataclass
class StubGeminiResponse:
    """Stub response for Gemini client interactions."""

    text: str
    model: str

    def model_dump(self) -> dict[str, object]:
        """Return stub raw payload."""
        return {"raw": True}


class StubGeminiModels:
    """Stub models resource capturing generate_content calls."""

    def __init__(self) -> None:
        """Prepare storage for captured kwargs."""
        self.kwargs: dict[str, object] | None = None

    def generate_content(self, **kwargs: object) -> StubGeminiResponse:
        """Capture generation arguments and return stub response."""
        self.kwargs = kwargs
        return StubGeminiResponse(text="gemini-image", model="gemini-2.0")


class StubGemini:
    """Stub Gemini client exposing models."""

    def __init__(self) -> None:
        """Expose stub models resource."""
        self.models = StubGeminiModels()


def test_gemini_vision_uses_inline_data() -> None:
    """Gemini vision calls inline binary data with the right mime type."""
    stub_client = StubGemini()
    client = GeminiLLMClient(cast("Any", stub_client), "gemini-default", temperature=0.4)
    image = ImageInput(media_type="image/webp", data="dmFsdWU=")

    response = client.vision("parse", image, system_prompt="s")

    assert response.content == "gemini-image"
    assert response.model == "gemini-2.0"
    assert stub_client.models.kwargs is not None
    parts = stub_client.models.kwargs["contents"][0]["parts"]  # type: ignore[index]
    assert parts[0]["text"] == "parse"  # type: ignore[index]
    inline_data = parts[1]["inline_data"]  # type: ignore[index]
    assert inline_data["mime_type"] == "image/webp"
    assert inline_data["data"] == image.data
    assert stub_client.models.kwargs.get("system_instruction") == "s"


def test_gemini_multimodal_supports_audio_and_video() -> None:
    """Gemini multimodal calls should inline audio and video payloads."""
    stub_client = StubGemini()
    client = GeminiLLMClient(cast("Any", stub_client), "gemini-default", temperature=0.2)
    audio = MediaInput(media_type="audio/wav", data="YXVkaW8=")
    video = MediaInput(media_type="video/mp4", data="dmlkZW8=")

    response = client.multimodal("describe", [audio, video], model="gemini-2.1", system_prompt="sys")

    assert response.content == "gemini-image"
    assert response.model == "gemini-2.0"
    assert stub_client.models.kwargs is not None
    parts = stub_client.models.kwargs["contents"][0]["parts"]  # type: ignore[index]
    assert parts[0]["text"] == "describe"  # type: ignore[index]
    inline_audio = parts[1]["inline_data"]  # type: ignore[index]
    inline_video = parts[2]["inline_data"]  # type: ignore[index]
    assert inline_audio["mime_type"] == "audio/wav"
    assert inline_audio["data"] == audio.data
    assert inline_video["mime_type"] == "video/mp4"
    assert inline_video["data"] == video.data
    assert stub_client.models.kwargs.get("system_instruction") == "sys"
