from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from agents import Agent
    from agents.items import TResponseInputItem
    from agents.result import RunResult
else:
    Agent = object
    TResponseInputItem = Any
    RunResult = Any

from various_llm_benchmark.agents.providers import AgnoAgentProvider, OpenAIAgentsProvider
from various_llm_benchmark.models import ImageInput


class StubRunResult:
    """Stub result object mimicking agent outputs."""

    def __init__(self, output: str, model: str) -> None:
        """Store stubbed content for assertions."""
        self.final_output = output
        self.new_items: list[object] = []
        self.content = output
        self.model = model
        self.__dataclass_fields__ = None


def test_openai_agents_provider_vision_builds_image_input() -> None:
    """OpenAI Agents provider should embed image URLs in run input."""
    captured: dict[str, object] = {}

    def run_function(agent: Agent[Any], run_input: str | list[TResponseInputItem]) -> RunResult:
        captured["agent"] = agent
        captured["input"] = run_input
        return cast("RunResult", StubRunResult("agent-vision", "gpt-4o"))

    provider = OpenAIAgentsProvider(
        api_key="key",
        model="gpt-4o",
        instructions="sys",
        temperature=0.2,
        run_function=run_function,
    )
    image = ImageInput(media_type="image/png", data="YmluYXJ5")

    response = provider.vision("prompt", image)

    assert response.content == "agent-vision"
    assert captured["input"]  # type: ignore[truthy-bool]
    payload = captured["input"][0]  # type: ignore[index]
    assert payload["content"][0]["text"] == "prompt"  # type: ignore[index]
    assert payload["content"][1]["image_url"].endswith(image.data)  # type: ignore[index]


@dataclass
class StubAgent:
    """Stub agent that records the provided run input."""

    model: object
    received: list[dict[str, object]] | None = None

    def run(
        self,
        run_input: list[dict[str, object]] | str,
        *,
        stream: bool | None = None,
        **_: object,
    ) -> StubRunResult:
        """Record provided run input without contacting any backend."""
        del stream
        if isinstance(run_input, list):
            converted: list[dict[str, object]] = list(run_input)
        else:
            converted = [{"content": run_input}]

        self.received = converted
        return StubRunResult("agno-vision", "agne-model")


def test_agno_agent_provider_vision_uses_data_url() -> None:
    """Agno provider should format image payloads as data URLs."""
    created_agents: list[StubAgent] = []

    def agent_factory(model: object) -> StubAgent:
        stub = StubAgent(model=model)
        created_agents.append(stub)
        return stub

    provider = AgnoAgentProvider(
        openai_api_key="key",
        anthropic_api_key="key",
        gemini_api_key="key",
        openai_model="gpt-openai",
        anthropic_model="claude",
        gemini_model="gemini",
        temperature=0.5,
        instructions="sys",
        agent_factory=agent_factory,
    )
    image = ImageInput(media_type="image/png", data="ZGF0YQ==")

    response = provider.vision("look", image, provider="openai")

    assert response.content == "agno-vision"
    assert created_agents  # type: ignore[truthy-bool]
    stub_agent = created_agents[-1]
    assert stub_agent.received is not None
    payload = stub_agent.received[0]
    assert payload["role"] == "user"  # type: ignore[index]
    assert payload["content"][0]["text"] == "look"  # type: ignore[index]
    assert payload["content"][1]["image_url"].endswith(image.data)  # type: ignore[index]
