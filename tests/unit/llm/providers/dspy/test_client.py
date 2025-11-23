from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from various_llm_benchmark.llm.providers.dspy.client import DsPyLLMClient, SupportsDsPyLM
from various_llm_benchmark.models import ChatMessage, LLMResponse

if TYPE_CHECKING:
    from litellm.types.utils import ModelResponse
else:
    ModelResponse = Any


class _MockDsPyLM(SupportsDsPyLM):
    model_type = "chat"

    def __init__(self, *, model: str, temperature: float, **kwargs: object) -> None:
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls.append({"prompt": prompt, "messages": messages, **kwargs})
        text = prompt if prompt is not None else "|".join(
            message["content"] for message in messages or []
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            model=self.model,
        )
        return cast("ModelResponse", response)

    def copy(self, **kwargs: Any) -> _MockDsPyLM:
        params: dict[str, object] = {**self.kwargs, **kwargs}
        model = cast("str", params.pop("model", self.model))
        temperature_value = params.pop("temperature", self.temperature)
        temperature = float(cast("float", temperature_value))
        return _MockDsPyLM(model=model, temperature=temperature, **params)


@pytest.fixture
def client_with_lms() -> tuple[DsPyLLMClient, list[_MockDsPyLM]]:
    """Provide a DsPy client and track constructed mock LMs."""
    created_lms: list[_MockDsPyLM] = []

    def factory(*, model: str, temperature: float, **kwargs: object) -> SupportsDsPyLM:
        lm = _MockDsPyLM(model=model, temperature=temperature, **kwargs)
        created_lms.append(lm)
        return lm

    client = DsPyLLMClient("dspy-base", temperature=0.3, lm_factory=factory)
    return client, created_lms


def test_generate_returns_llm_response(client_with_lms: tuple[DsPyLLMClient, list[_MockDsPyLM]]) -> None:
    """Generate should return an LLMResponse built from the mock LM output."""
    client, created_lms = client_with_lms

    response = client.generate("hello")

    assert isinstance(response, LLMResponse)
    assert response.content == "hello"
    assert response.model == "dspy-base"
    assert response.raw["outputs"] == ["hello"]
    assert created_lms[0].calls == [{"prompt": "hello", "messages": None}]


def test_chat_returns_llm_response(client_with_lms: tuple[DsPyLLMClient, list[_MockDsPyLM]]) -> None:
    """Chat should return an LLMResponse using formatted chat messages."""
    client, created_lms = client_with_lms
    messages = [
        ChatMessage(role="system", content="s"),
        ChatMessage(role="user", content="hi"),
    ]

    response = client.chat(messages)

    assert isinstance(response, LLMResponse)
    assert response.content == "s|hi"
    assert response.model == "dspy-base"
    assert response.raw["outputs"] == ["s|hi"]
    assert created_lms[0].calls == [
        {
            "prompt": None,
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "hi"},
            ],
        },
    ]
