from __future__ import annotations

from typing import TYPE_CHECKING

from various_llm_benchmark.settings import Settings

if TYPE_CHECKING:
    import pytest


def _required_keys() -> dict[str, str]:
    return {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "GEMINI_API_KEY": "test-gemini-key",
    }


def test_settings_includes_dspy_models_defaults() -> None:
    """Ensure default DsPy model values are loaded."""
    settings = Settings.model_validate(_required_keys())

    assert settings.dspy_model == "gpt-5.1"
    assert settings.dspy_light_model == "gpt-5.1-mini"


def test_settings_overrides_dspy_models_via_env() -> None:
    """Allow overriding DsPy model values via environment variables."""
    env = {
        **_required_keys(),
        "DSPY_MODEL": "custom-dspy",
        "DSPY_LIGHT_MODEL": "custom-dspy-light",
    }

    settings = Settings.model_validate(env)

    assert settings.dspy_model == "custom-dspy"
    assert settings.dspy_light_model == "custom-dspy-light"


def test_settings_include_pydantic_ai_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure Pydantic AI settings expose sensible defaults."""
    monkeypatch.delenv("PYDANTIC_AI_API_KEY", raising=False)
    settings = Settings.model_validate(_required_keys())

    assert settings.pydantic_ai_model == "gpt-5.1"
    assert settings.pydantic_ai_light_model == "gpt-5.1-mini"
    assert settings.pydantic_ai_api_key.get_secret_value() == ""


def test_settings_overrides_pydantic_ai_via_env() -> None:
    """Allow overriding Pydantic AI settings through environment variables."""
    env = {
        **_required_keys(),
        "PYDANTIC_AI_API_KEY": "pydantic-api-key",
        "PYDANTIC_AI_MODEL": "pydantic-model",
        "PYDANTIC_AI_LIGHT_MODEL": "pydantic-light-model",
    }

    settings = Settings.model_validate(env)

    assert settings.pydantic_ai_api_key.get_secret_value() == "pydantic-api-key"
    assert settings.pydantic_ai_model == "pydantic-model"
    assert settings.pydantic_ai_light_model == "pydantic-light-model"
