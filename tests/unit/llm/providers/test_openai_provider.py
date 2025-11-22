from __future__ import annotations

from typing import TYPE_CHECKING
from types import SimpleNamespace

from various_llm_benchmark.llm.providers.openai.client import OpenAILLMClient
from various_llm_benchmark.models import ChatMessage

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_generate_calls_openai(mocker: MockerFixture) -> None:
    """OpenAI completion should receive prompt and return parsed response."""
    mock_output = [SimpleNamespace(content=[SimpleNamespace(text=SimpleNamespace(value="hello"))])]
    mock_completion = mocker.Mock()
    mock_completion.output = mock_output
    mock_completion.model = "gpt-test"
    mock_completion.model_dump.return_value = {"model": "gpt-test"}

    mock_client = mocker.Mock()
    mock_client.responses.create.return_value = mock_completion

    client = OpenAILLMClient(mock_client, "gpt-default", temperature=0.4)
    response = client.generate("hi")

    mock_client.responses.create.assert_called_once_with(
        model="gpt-default", input="hi", temperature=0.4,
    )
    assert response.content == "hello"
    assert response.model == "gpt-test"
    assert response.raw == {"model": "gpt-test"}


def test_chat_sends_messages(mocker: MockerFixture) -> None:
    """OpenAI chat should forward history and return content."""
    mock_output = [SimpleNamespace(content=[SimpleNamespace(text=SimpleNamespace(value="answer"))])]
    mock_completion = mocker.Mock()
    mock_completion.output = mock_output
    mock_completion.model = "gpt-2"
    mock_completion.model_dump.return_value = {"model": "gpt-2"}

    mock_client = mocker.Mock()
    mock_client.responses.create.return_value = mock_completion

    client = OpenAILLMClient(mock_client, "gpt-default")
    history = [ChatMessage(role="system", content="be nice"), ChatMessage(role="user", content="hello")]
    response = client.chat(history, model="gpt-1")

    mock_client.responses.create.assert_called_once_with(
        model="gpt-1",
        input=[{"role": "system", "content": "be nice"}, {"role": "user", "content": "hello"}],
        temperature=0.7,
    )
    assert response.content == "answer"
    assert response.model == "gpt-2"
