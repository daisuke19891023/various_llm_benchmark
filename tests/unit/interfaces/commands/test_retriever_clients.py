"""Tests for retriever command helpers."""

from __future__ import annotations

import pytest

from various_llm_benchmark.interfaces.commands import retriever_clients
from various_llm_benchmark.llm.tools.retriever import RetrieverError


def test_retrieve_raises_when_no_backends_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid DB usage and return a clear error when search backends are disabled."""
    monkeypatch.setattr(retriever_clients.settings, "enable_pgvector", False)
    monkeypatch.setattr(retriever_clients.settings, "enable_pgroonga", False)

    pool_called = False

    def _fake_create_pool() -> object:
        nonlocal pool_called
        pool_called = True
        return object()

    monkeypatch.setattr(retriever_clients, "create_postgres_pool", _fake_create_pool)

    executor = retriever_clients.resolve_retriever_client("openai")

    with pytest.raises(RetrieverError, match="No retriever backends are enabled"):
        executor("query")

    assert pool_called is False
