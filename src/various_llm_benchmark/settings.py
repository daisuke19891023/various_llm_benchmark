from __future__ import annotations

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    openai_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="ANTHROPIC_API_KEY")
    gemini_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="GEMINI_API_KEY")
    openai_model: str = Field(default="gpt-5.1", validation_alias="OPENAI_MODEL")
    openai_light_model: str = Field(default="gpt-5.1-mini", validation_alias="OPENAI_LIGHT_MODEL")
    anthropic_model: str = Field(default="claude-4.5-sonnet", validation_alias="ANTHROPIC_MODEL")
    anthropic_light_model: str = Field(
        default="claude-4.5-haiku", validation_alias="ANTHROPIC_LIGHT_MODEL",
    )
    gemini_model: str = Field(default="gemini-3.0-pro", validation_alias="GEMINI_MODEL")
    gemini_light_model: str = Field(default="gemini-2.5-flash", validation_alias="GEMINI_LIGHT_MODEL")
    dspy_model: str = Field(default="gpt-5.1", validation_alias="DSPY_MODEL")
    dspy_light_model: str = Field(default="gpt-5.1-mini", validation_alias="DSPY_LIGHT_MODEL")
    default_temperature: float = Field(default=0.7, validation_alias="DEFAULT_TEMPERATURE")
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @model_validator(mode="after")
    def validate_keys(self) -> Settings:
        """Ensure mandatory API keys are provided."""
        missing_keys: list[str] = []
        if not self.openai_api_key.get_secret_value():
            missing_keys.append("OPENAI_API_KEY")
        if not self.anthropic_api_key.get_secret_value():
            missing_keys.append("ANTHROPIC_API_KEY")
        if not self.gemini_api_key.get_secret_value():
            missing_keys.append("GEMINI_API_KEY")
        if missing_keys:
            missing = ", ".join(missing_keys)
            error_message = f"Missing required keys: {missing}"
            raise ValueError(error_message)
        return self


settings = Settings()
