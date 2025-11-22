from __future__ import annotations

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    openai_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="ANTHROPIC_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022", validation_alias="ANTHROPIC_MODEL")
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
        if missing_keys:
            missing = ", ".join(missing_keys)
            error_message = f"Missing required keys: {missing}"
            raise ValueError(error_message)
        return self


settings = Settings()
