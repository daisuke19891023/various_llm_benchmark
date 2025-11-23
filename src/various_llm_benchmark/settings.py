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
    postgres_connection_string: SecretStr = Field(
        default=SecretStr(""), validation_alias="POSTGRES_CONNECTION_STRING",
    )
    postgres_schema: str = Field(default="", validation_alias="POSTGRES_SCHEMA")
    pgvector_table_name: str = Field(default="", validation_alias="PGVECTOR_TABLE_NAME")
    pgroonga_table_name: str = Field(default="", validation_alias="PGROONGA_TABLE_NAME")
    enable_pgvector: bool = Field(default=False, validation_alias="ENABLE_PGVECTOR")
    enable_pgroonga: bool = Field(default=False, validation_alias="ENABLE_PGROONGA")
    search_top_k: int = Field(default=5, validation_alias="SEARCH_TOP_K", ge=1)
    search_score_threshold: float = Field(
        default=0.0, validation_alias="SEARCH_SCORE_THRESHOLD", ge=0.0, le=1.0,
    )
    embedding_model: str = Field(default="", validation_alias="EMBEDDING_MODEL")
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @model_validator(mode="after")
    def validate_keys(self) -> Settings:
        """Ensure mandatory API keys are provided."""
        missing_keys: list[str] = [
            key
            for key, value in (
                ("OPENAI_API_KEY", self.openai_api_key.get_secret_value()),
                ("ANTHROPIC_API_KEY", self.anthropic_api_key.get_secret_value()),
                ("GEMINI_API_KEY", self.gemini_api_key.get_secret_value()),
            )
            if not value
        ]

        if self.enable_pgvector or self.enable_pgroonga:
            missing_keys.extend(
                key
                for key, value in (
                    (
                        "POSTGRES_CONNECTION_STRING",
                        self.postgres_connection_string.get_secret_value(),
                    ),
                    ("POSTGRES_SCHEMA", self.postgres_schema),
                    ("EMBEDDING_MODEL", self.embedding_model),
                )
                if not value
            )

            feature_specific_requirements: list[tuple[str, str]] = []
            if self.enable_pgvector:
                feature_specific_requirements.append(
                    ("PGVECTOR_TABLE_NAME", self.pgvector_table_name),
                )
            if self.enable_pgroonga:
                feature_specific_requirements.append(
                    ("PGROONGA_TABLE_NAME", self.pgroonga_table_name),
                )

            missing_keys.extend(
                [key for key, value in feature_specific_requirements if not value],
            )

        if missing_keys:
            missing = ", ".join(missing_keys)
            error_message = f"Missing required keys: {missing}"
            raise ValueError(error_message)
        return self


settings = Settings()
