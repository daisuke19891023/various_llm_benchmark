"""Unit tests for retriever ingestion utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast

from psycopg import Error as PsycopgError

from various_llm_benchmark.llm.retriever_ingest import (
    BaseChunker,
    ChunkerStrategy,
    DocumentChunk,
    SlidingWindowChunker,
    SupportsCursor,
    SupportsTransactionalConnection,
    TextDocument,
    chunk_documents,
    ingest_text_directory,
    load_text_documents,
)
from various_llm_benchmark.settings import settings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from _pytest.monkeypatch import MonkeyPatch


class RecordingCursor:
    """Cursor stub that can fail once then succeed."""

    def __init__(self, *, fail_once: bool = False) -> None:
        """Store the failure toggle."""
        self.fail_once = fail_once
        self.calls: list[tuple[object, list[object]]] = []

    def execute(self, query: object, params: Sequence[object] | None = None) -> None:
        """Record executions and optionally raise a transient error."""
        if self.fail_once:
            self.fail_once = False
            raise PsycopgError("temporary error")
        self.calls.append((query, list(params or [])))

    def __enter__(self) -> Self:
        """Enable use as a context manager."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        """No-op context manager exit."""
        return False


class RecordingConnection:
    """Connection stub tracking commits and rollbacks."""

    def __init__(self, cursor: RecordingCursor) -> None:
        """Store the cursor and initialize counters."""
        self.cursor_instance = cursor
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> SupportsCursor:
        """Return the attached cursor."""
        return self.cursor_instance

    def commit(self) -> None:
        """Increment commit counter."""
        self.commits += 1

    def rollback(self) -> None:
        """Increment rollback counter."""
        self.rollbacks += 1

    def __enter__(self) -> Self:
        """Enable use as a context manager."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> object:
        """No-op context manager exit."""
        return False


class RecordingPool:
    """Pool stub returning a preconstructed connection."""

    def __init__(self, connection: RecordingConnection) -> None:
        """Store the connection instance."""
        self.connection_instance = connection
        self.timeouts: list[float | None] = []

    def connection(self, timeout: float | None = None) -> SupportsTransactionalConnection:
        """Return the stored connection and record timeout."""
        self.timeouts.append(timeout)
        return self.connection_instance


def test_load_and_chunk_sliding_window(tmp_path: Path) -> None:
    """Text files should be loaded and chunked via the sliding window strategy."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("abcdef", encoding="utf-8")
    (docs_dir / "b.txt").write_text("hello world", encoding="utf-8")

    documents = load_text_documents(docs_dir)
    assert [Path(doc.path).name for doc in documents] == ["a.txt", "b.txt"]

    chunker = SlidingWindowChunker(chunk_size=4, overlap=1)
    chunks = chunk_documents(documents, chunker)

    assert len(chunks) == 6
    first_chunk = chunks[0]
    assert first_chunk.id.endswith("a.txt::chunk-0")
    assert first_chunk.metadata["chunk_start"] == 0
    assert first_chunk.metadata["chunk_end"] == 4


def test_ingest_text_directory_retries_and_upserts(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ingestion should retry failed upserts and commit successfully."""
    monkeypatch.setattr(settings, "enable_pgvector", True)
    monkeypatch.setattr(settings, "postgres_schema", "public")
    monkeypatch.setattr(settings, "pgvector_table_name", "items")

    def _sleep(_: float) -> None:
        return None

    monkeypatch.setattr("time.sleep", _sleep)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "doc.txt").write_text("first block\n\nsecond block", encoding="utf-8")

    cursor = RecordingCursor(fail_once=True)
    connection = RecordingConnection(cursor)
    pool = RecordingPool(connection)

    embedded_chunks: list[float] = []

    def _embed(chunk: DocumentChunk) -> list[float]:
        embedded_chunks.append(float(len(chunk.content)))
        return [float(len(chunk.content))]

    ingest_text_directory(
        docs_dir,
        pool=pool,
        chunk_strategy=ChunkerStrategy.PARAGRAPH,
        chunker_options={"min_length": 3},
        embedding_fn=_embed,
        max_retries=2,
        batch_size=5,
        timeout=0.0,
    )

    assert connection.commits == 1
    assert connection.rollbacks == 1
    assert pool.timeouts[-1] == 0.0

    assert len(cursor.calls) == 1
    metadata = json.loads(str(cursor.calls[0][1][2]))
    assert metadata["name"] == "doc.txt"
    assert embedded_chunks == [len("first block"), len("second block")]


def test_ingest_text_directory_uses_chunk_strategy_name(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Chunk strategy names should resolve via the registry mapping."""
    monkeypatch.setattr(settings, "enable_pgvector", True)
    monkeypatch.setattr(settings, "postgres_schema", "public")
    monkeypatch.setattr(settings, "pgvector_table_name", "items")

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "doc.txt").write_text("alpha\n\nbeta", encoding="utf-8")

    cursor = RecordingCursor()
    connection = RecordingConnection(cursor)
    pool = RecordingPool(connection)

    class CustomChunker(BaseChunker):
        """Chunker returning one chunk per document."""

        def chunk(self, document: TextDocument) -> list[DocumentChunk]:
            return [DocumentChunk(id=f"custom::{document.path}", content=document.content, metadata={"custom": True})]

    recorded_values: dict[str, object] = {}

    def _capture_execute_values(
        cursor_obj: RecordingCursor,
        query: object,
        values: Sequence[tuple[object, ...]],
        template: str | None = None,
    ) -> None:
        recorded_values["cursor"] = cursor_obj
        recorded_values["query"] = query
        recorded_values["values"] = list(values)
        recorded_values["template"] = template

    monkeypatch.setattr("various_llm_benchmark.llm.retriever_ingest.execute_values", _capture_execute_values)

    ingest_text_directory(
        docs_dir,
        pool=pool,
        chunk_strategy="custom",
        chunker_registry={"custom": CustomChunker},
        embedding_fn=lambda chunk: [float(len(chunk.content))],
        timeout=0.0,
    )

    assert recorded_values["cursor"] is cursor
    values = cast("list[tuple[object, object, object, object]]", recorded_values["values"])
    assert len(values) == 1
    chunk_id, _, metadata, _ = values[0]
    assert cast("str", chunk_id).startswith("custom::")
    assert json.loads(str(metadata))["custom"] is True


def test_ingest_text_directory_bulk_inserts(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Embeddings should be inserted via a single batched call."""
    monkeypatch.setattr(settings, "enable_pgvector", True)
    monkeypatch.setattr(settings, "postgres_schema", "public")
    monkeypatch.setattr(settings, "pgvector_table_name", "items")

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "doc.txt").write_text("first paragraph\n\nsecond paragraph", encoding="utf-8")

    cursor = RecordingCursor()
    connection = RecordingConnection(cursor)
    pool = RecordingPool(connection)

    recorded_values: dict[str, object] = {}

    def _capture_execute_values(
        cursor_obj: RecordingCursor,
        query: object,
        values: Sequence[tuple[object, ...]],
        template: str | None = None,
    ) -> None:
        recorded_values["cursor"] = cursor_obj
        recorded_values["query"] = query
        recorded_values["values"] = list(values)
        recorded_values["template"] = template

    monkeypatch.setattr("various_llm_benchmark.llm.retriever_ingest.execute_values", _capture_execute_values)

    ingest_text_directory(
        docs_dir,
        pool=pool,
        chunk_strategy="paragraph",
        embedding_fn=lambda chunk: [float(len(chunk.content))],
        timeout=0.0,
    )

    assert recorded_values["cursor"] is cursor
    assert recorded_values["template"] == "(%s, %s, %s::jsonb, %s::vector)"
    values = cast("list[tuple[object, object, object, object]]", recorded_values["values"])
    assert len(values) == 2
    first_metadata = json.loads(str(values[0][2]))
    assert first_metadata["chunk_index"] == 0
