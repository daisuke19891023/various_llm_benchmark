from __future__ import annotations

from typing import TYPE_CHECKING

from various_llm_benchmark.llm.protocol import ChatMessage
from various_llm_benchmark.llm.providers.anthropic.client import AnthropicLLMClient

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_generate_calls_anthropic(mocker: MockerFixture) -> None:
    """Anthropic completion should receive prompt and parse response."""
    mock_response = mocker.Mock()
    mock_response.content = ["hi"]
    mock_response.model = "claude-test"
    mock_response.model_dump.return_value = {"model": "claude-test"}

    mock_client = mocker.Mock()
    mock_client.messages.create.return_value = mock_response

    client = AnthropicLLMClient(mock_client, "claude-default", temperature=0.3)
    response = client.generate("hello")

    mock_client.messages.create.assert_called_once_with(
        model="claude-default",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=1024,
        temperature=0.3,
    )
    assert response.content == "hi"
    assert response.model == "claude-test"
    assert response.raw == {"model": "claude-test"}


def test_chat_with_history(mocker: MockerFixture) -> None:
    """Anthropic chat should forward history and return content."""
    mock_response = mocker.Mock()
    mock_response.content = [{"text": "いいよ"}]
    mock_response.model = "claude-1"
    mock_response.model_dump.return_value = {"model": "claude-1"}

    mock_client = mocker.Mock()
    mock_client.messages.create.return_value = mock_response

    client = AnthropicLLMClient(mock_client, "claude-default")
    history = [ChatMessage(role="user", content="元気?")]
    response = client.chat(history)

    mock_client.messages.create.assert_called_once_with(
        model="claude-default",
        messages=[{"role": "user", "content": "元気?"}],
        max_tokens=1024,
        temperature=0.7,
    )
    assert response.content == "いいよ"
    assert response.model == "claude-1"
