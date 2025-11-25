from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

    from litellm.types.utils import ModelResponse

from various_llm_benchmark.llm.providers.dspy.client import DsPyLLMClient, SupportsDsPyLM
from various_llm_benchmark.models import ChatMessage, ImageInput


class _DummyLM(SupportsDsPyLM):
    model: str
    model_type: str

    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        max_tokens: int = 1000,
        cache: bool = True,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.model_type = "chat"
        self.kwargs: dict[str, object] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "cache": cache,
            **kwargs,
        }
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls.append({"prompt": prompt, "messages": messages, **kwargs})
        if messages is not None:
            completion = "|".join(message["content"] for message in messages)
            message_obj = SimpleNamespace(content=completion)
        else:
            message_obj = SimpleNamespace(content=f"echo:{prompt}")
        choice = SimpleNamespace(message=message_obj)
        response = SimpleNamespace(
            choices=[choice],
            model=self.model,
            usage={},
            _hidden_params={},
        )
        return cast("ModelResponse", response)

    def copy(self, **kwargs: Any) -> _DummyLM:
        updated_kwargs: dict[str, object] = {**self.kwargs, **kwargs}
        model = cast("str", updated_kwargs.pop("model", self.model))
        temperature_value = updated_kwargs.pop("temperature", self.kwargs["temperature"])
        temperature = float(cast("float", temperature_value))
        max_tokens_value = updated_kwargs.pop("max_tokens", self.kwargs.get("max_tokens", 1000))
        cache_value = updated_kwargs.pop("cache", self.kwargs.get("cache", True))
        max_tokens = int(cast("int", max_tokens_value))
        cache = bool(cast("bool", cache_value))
        return _DummyLM(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            **updated_kwargs,
        )


def test_generate_calls_dspy_lm() -> None:
    """DsPy generate should call LM with prompt and return normalized response."""
    created_lms: list[_DummyLM] = []

    def factory(*, model: str, temperature: float, **kwargs: Any) -> SupportsDsPyLM:
        lm = _DummyLM(model=model, temperature=temperature, **kwargs)
        created_lms.append(lm)
        return lm

    client = DsPyLLMClient("dspy-default", temperature=0.2, lm_factory=factory, top_p=0.9)
    response = client.generate("hello")

    assert created_lms
    lm = created_lms[0]
    assert lm.model == "dspy-default"
    assert lm.kwargs["temperature"] == 0.2
    assert lm.kwargs["top_p"] == 0.9
    assert lm.calls == [{"prompt": "hello", "messages": None}]
    assert response.content == "echo:hello"
    assert response.model == "dspy-default"
    assert response.raw["outputs"] == ["echo:hello"]
    raw_response = cast("dict[str, object]", response.raw["response"])
    choices = cast("list[object]", raw_response["choices"])
    first_choice = cast("Mapping[str, object]", choices[0])
    message = cast("Mapping[str, object]", first_choice["message"])
    assert cast("str", message["content"]) == "echo:hello"


def test_chat_uses_model_override() -> None:
    """DsPy chat should format messages and respect explicit model override."""
    created_lms: list[_DummyLM] = []

    def factory(*, model: str, temperature: float, **kwargs: Any) -> SupportsDsPyLM:
        lm = _DummyLM(model=model, temperature=temperature, **kwargs)
        created_lms.append(lm)
        return lm

    client = DsPyLLMClient("dspy-default", lm_factory=factory)
    history = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="assistant", content="ok"),
        ChatMessage(role="user", content="hi"),
    ]

    response = client.chat(history, model="dspy-override")

    assert len(created_lms) == 2  # default during init + override
    override_lm = created_lms[1]
    assert override_lm.model == "dspy-override"
    assert override_lm.calls == [
        {
            "prompt": None,
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "hi"},
            ],
        },
    ]
    assert response.content == "sys|ok|hi"
    assert response.model == "dspy-override"
    assert response.raw["outputs"] == ["sys|ok|hi"]
    raw_response = cast("dict[str, object]", response.raw["response"])
    choices = cast("list[object]", raw_response["choices"])
    first_choice = cast("Mapping[str, object]", choices[0])
    message = cast("Mapping[str, object]", first_choice["message"])
    assert cast("str", message["content"]) == "sys|ok|hi"


def test_vision_not_supported() -> None:
    """Vision should raise NotImplementedError until supported."""

    def factory(*, model: str, temperature: float, **kwargs: Any) -> SupportsDsPyLM:
        return _DummyLM(model=model, temperature=temperature, **kwargs)

    client = DsPyLLMClient("dspy-default", lm_factory=factory)

    try:
        client.vision("prompt", image=ImageInput(media_type="image/png", data="d"))
    except NotImplementedError:
        return
    raise AssertionError("Expected NotImplementedError for vision")
