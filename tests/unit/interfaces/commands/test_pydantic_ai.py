from __future__ import annotations

import os
from typing import Any

from pydantic import SecretStr
from typer.testing import CliRunner

from various_llm_benchmark.interfaces.commands import pydantic_ai
from various_llm_benchmark.models import LLMResponse


def test_create_provider_uses_settings(monkeypatch: Any) -> None:
    """Factory constructs provider with settings-derived defaults."""
    created: dict[str, object] = {}

    class DummyProvider:
        def __init__(
            self, *, model: str, system_prompt: str | None, temperature: float,
        ) -> None:
            created.update(model=model, system_prompt=system_prompt, temperature=temperature)

    monkeypatch.setattr(pydantic_ai, "PydanticAIAgentProvider", DummyProvider)
    monkeypatch.setattr(pydantic_ai.settings, "openai_model", "primary-model")
    monkeypatch.setattr(pydantic_ai.settings, "openai_light_model", "light-model")
    monkeypatch.setattr(pydantic_ai.settings, "default_temperature", 0.35)
    monkeypatch.setattr(pydantic_ai.settings, "openai_api_key", SecretStr("factory-key"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = pydantic_ai.create_provider()

    assert isinstance(provider, DummyProvider)
    assert created["model"] == "primary-model"
    assert created["system_prompt"]
    assert created["temperature"] == 0.35
    assert os.environ["OPENAI_API_KEY"] == "factory-key"

    light_provider = pydantic_ai.create_provider(use_light_model=True)

    assert isinstance(light_provider, DummyProvider)
    assert created["model"] == "light-model"


def test_complete_command_invokes_provider(monkeypatch: Any) -> None:
    """Typer command delegates to provider and prints response."""
    runner = CliRunner()

    class DummyProvider:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def complete(self, prompt: str) -> LLMResponse:
            self.calls.append(prompt)
            return LLMResponse(content="ok", model="test", raw={})

    dummy_provider = DummyProvider()

    def fake_provider_creator(**_: object) -> DummyProvider:
        return dummy_provider

    monkeypatch.setattr(
        pydantic_ai,
        "create_provider",
        fake_provider_creator,
    )

    result = runner.invoke(pydantic_ai.pydantic_ai_app, ["complete", "hello"])

    assert result.exit_code == 0
    assert "ok" in result.stdout
    assert dummy_provider.calls == ["hello"]
