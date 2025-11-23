"""Settings validation tests."""

import pytest

from various_llm_benchmark.settings import Settings

def _set_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("VOYAGE_API_KEY", "voyage-key")


def test_validate_keys_requires_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """APIキーが空の場合に検証が失敗すること."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("VOYAGE_API_KEY", "")

    with pytest.raises(ValueError, match="OPENAI_API_KEY") as exc_info:
        Settings()

    message = str(exc_info.value)
    assert "GEMINI_API_KEY" in message


def test_validate_keys_allows_pgvector_without_voyage_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Voyage APIキーがなくてもpgvectorを利用できること."""
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("VOYAGE_API_KEY", "")
    monkeypatch.setenv("ENABLE_PGVECTOR", "true")
    monkeypatch.setenv("POSTGRES_CONNECTION_STRING", "postgresql://user:pass@localhost:5432/db")
    monkeypatch.setenv("POSTGRES_SCHEMA", "public")
    monkeypatch.setenv("PGVECTOR_TABLE_NAME", "documents")

    settings = Settings()

    assert settings.enable_pgvector is True


def test_validate_keys_require_postgres_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """pgvector有効時にPostgreSQL関連のキーが必須になること."""
    _set_api_keys(monkeypatch)
    monkeypatch.setenv("ENABLE_PGVECTOR", "true")

    with pytest.raises(ValueError, match="POSTGRES_CONNECTION_STRING") as exc_info:
        Settings()

    message = str(exc_info.value)
    assert "POSTGRES_SCHEMA" in message
    assert "PGVECTOR_TABLE_NAME" in message
    assert "EMBEDDING_MODEL" not in message


def test_validate_keys_accepts_valid_postgres_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """必要なPostgreSQL設定が揃っていれば検証に成功すること."""
    _set_api_keys(monkeypatch)
    monkeypatch.setenv("ENABLE_PGVECTOR", "true")
    monkeypatch.setenv("POSTGRES_CONNECTION_STRING", "postgresql://user:pass@localhost:5432/db")
    monkeypatch.setenv("POSTGRES_SCHEMA", "public")
    monkeypatch.setenv("PGVECTOR_TABLE_NAME", "documents")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("SEARCH_TOP_K", "10")
    monkeypatch.setenv("SEARCH_SCORE_THRESHOLD", "0.2")

    settings = Settings()

    assert settings.pgvector_table_name == "documents"
