"""Unit tests for the retriever utilities."""

from __future__ import annotations

import httpx
import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, Self

import pytest
from openai import APITimeoutError
from psycopg_pool import PoolTimeout

from various_llm_benchmark.llm.tools.retriever import (
    EmbeddingProvider,
    RetrievedDocument,
    RetrieverError,
    SupportsEmbeddingCreate,
    generate_embedding,
    merge_ranked_results,
    pgroonga_full_text_search,
    pgvector_similarity_search,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class FakeCursor:
    """Cursor stub that records executed queries and returns preset rows."""

    def __init__(self, rows: Sequence[tuple[object, ...]]) -> None:
        """Store the rows to be returned by ``fetchall``."""
        self.rows = list(rows)
        self.executed: list[tuple[object, list[object]]] = []

    def execute(self, query: object, params: Iterable[object]) -> None:
        """Record the incoming query and parameters."""
        self.executed.append((query, list(params)))

    def fetchall(self) -> Sequence[tuple[object, ...]]:
        """Return the preset rows."""
        return self.rows

    def __enter__(self) -> Self:
        """Enable use as a context manager."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        """No-op context manager exit."""
        return False


class FakeConnection:
    """Connection stub for wrapping a fake cursor."""

    def __init__(self, cursor: FakeCursor) -> None:
        """Store the cursor instance."""
        self.cursor_instance = cursor

    def cursor(self) -> FakeCursor:
        """Return the stored cursor."""
        return self.cursor_instance

    def __enter__(self) -> Self:
        """Enable use as a context manager."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        """No-op context manager exit."""
        return False


class FakePool:
    """Pool stub to simulate psycopg connection pools."""

    def __init__(self, cursor: FakeCursor | None = None, *, error: Exception | None = None) -> None:
        """Store either a cursor for success cases or an error to raise."""
        self.cursor = cursor
        self.error = error
        self.last_timeout: float | None = None

    def connection(self, timeout: float | None = None) -> FakeConnection:
        """Return a fake connection or raise the configured error."""
        self.last_timeout = timeout
        if self.error is not None:
            raise self.error
        assert self.cursor is not None
        return FakeConnection(self.cursor)


class FlakyEmbeddings:
    """Embedding stub that fails once before succeeding."""

    def __init__(self) -> None:
        """Initialize the call counter."""
        self.calls = 0

    def create(self, **_: object) -> SimpleNamespace:
        """Fail on the first call and succeed afterwards."""
        self.calls += 1
        if self.calls == 1:
            dummy_request = httpx.Request("GET", "https://example.com")
            raise APITimeoutError(request=dummy_request)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.2, 0.3])])


class FakeOpenAIClient:
    """OpenAI client stub exposing an embeddings attribute."""

    def __init__(self, embeddings: FlakyEmbeddings) -> None:
        """Store the embeddings stub."""
        self.embeddings: SupportsEmbeddingCreate = embeddings


def test_generate_embedding_retries_on_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI embedding generation should retry transient failures."""
    def _sleep(_: float) -> None:
        return None

    monkeypatch.setattr("various_llm_benchmark.llm.tools.retriever.time.sleep", _sleep)
    embeddings = FlakyEmbeddings()
    client = FakeOpenAIClient(embeddings)

    vector = generate_embedding(
        "hello",
        provider=EmbeddingProvider.OPENAI,
        model="text-embed",
        openai_client=client,
        max_retries=1,
        timeout=0.0,
    )

    assert vector == [0.2, 0.3]
    assert embeddings.calls == 2


def test_pgvector_similarity_search_filters_by_threshold() -> None:
    """Pgvector search should respect similarity threshold and parameters."""
    cursor = FakeCursor(rows=[("1", "doc", {"lang": "en"}, 0.9), ("2", "other", None, 0.1)])
    pool = FakePool(cursor)

    results = pgvector_similarity_search(
        pool,
        [0.1, 0.2],
        schema="public",
        table="vectors",
        top_k=3,
        threshold=0.5,
        timeout=0.0,
    )

    assert [result.id for result in results] == ["1"]
    executed_query, params = cursor.executed[0]
    query_repr = str(executed_query)
    assert "vectors" in query_repr
    assert params[0] == [0.1, 0.2]
    assert params[1] == [0.1, 0.2]
    assert params[-1] == 3


def test_pgroonga_full_text_search_propagates_errors() -> None:
    """PGroonga search should wrap pool failures in RetrieverError."""
    pool = FakePool(error=PoolTimeout("timed out"))

    with pytest.raises(RetrieverError):
        pgroonga_full_text_search(pool, "hello", schema="public", table="documents", timeout=0.0)


def test_merge_ranked_results_normalizes_and_weights() -> None:
    """Normalization should rescale per source before weighting and merging."""
    vector_results = [
        RetrievedDocument(id="1", content="a", metadata={}, score=0.9, source="pgvector"),
        RetrievedDocument(id="2", content="b", metadata={}, score=0.4, source="pgvector"),
    ]
    text_results = [RetrievedDocument(id="1", content="a", metadata={}, score=2.0, source="pgroonga")]

    merged = merge_ranked_results(vector_results, text_results, vector_weight=0.7, text_weight=0.3)

    assert [item.id for item in merged] == ["1", "2"]
    assert merged[0].metadata["sources"] == ["pgroonga", "pgvector"]
    assert math.isclose(merged[0].score, 1.0)
    assert math.isclose(merged[1].score, 0.0)
