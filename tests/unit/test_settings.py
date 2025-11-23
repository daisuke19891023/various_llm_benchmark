from __future__ import annotations

from various_llm_benchmark.settings import Settings


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
