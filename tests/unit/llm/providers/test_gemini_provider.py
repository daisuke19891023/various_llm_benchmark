from __future__ import annotations

from typing import TYPE_CHECKING
from types import SimpleNamespace

from various_llm_benchmark.llm.providers.gemini.client import GeminiLLMClient
from various_llm_benchmark.models import ChatMessage

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_generate_calls_gemini(mocker: MockerFixture) -> None:
    """Gemini completion should receive prompt and parse response."""
    mock_response = SimpleNamespace(text="hello", model="gemini-pro")
    mock_response.model_dump = mocker.Mock(return_value={"model": "gemini-pro"})

    mock_models = mocker.Mock()
    mock_models.generate_content.return_value = mock_response

    mock_client = mocker.Mock()
    mock_client.models = mock_models

    client = GeminiLLMClient(mock_client, "gemini-3.0-pro", temperature=0.2)
    response = client.generate("hi")

    mock_models.generate_content.assert_called_once_with(
        model="gemini-3.0-pro", contents="hi", config={"temperature": 0.2},
    )
    assert response.content == "hello"
    assert response.model == "gemini-pro"
    assert response.raw == {"model": "gemini-pro"}


def test_chat_sends_history(mocker: MockerFixture) -> None:
    """Gemini chat should forward history and return content."""
    mock_response = SimpleNamespace(content={"parts": ["answer"]}, model="gemini-light")
    mock_response.model_dump = mocker.Mock(return_value={"model": "gemini-light"})

    mock_models = mocker.Mock()
    mock_models.generate_content.return_value = mock_response

    mock_client = mocker.Mock()
    mock_client.models = mock_models

    client = GeminiLLMClient(mock_client, "gemini-3.0-pro")
    history = [ChatMessage(role="system", content="guide"), ChatMessage(role="user", content="hello")]
    response = client.chat(history, model="gemini-2.5-flash")

    mock_models.generate_content.assert_called_once_with(
        model="gemini-2.5-flash",
        contents=[{"role": "system", "parts": ["guide"]}, {"role": "user", "parts": ["hello"]}],
        config={"temperature": 0.7},
    )
    assert response.content == "answer"
    assert response.model == "gemini-light"
