"""Utilities for ingesting text files into pgvector with configurable chunking."""

from __future__ import annotations

import importlib
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self, cast

from psycopg import Error as PsycopgError
from psycopg import sql
from pydantic import BaseModel, Field

from various_llm_benchmark.llm.tools.retriever import (
    EmbeddingProvider,
    SupportsGoogleClient,
    SupportsOpenAIClient,
    SupportsVoyageClient,
    generate_embeddings_batch,
)
from various_llm_benchmark.settings import settings

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


_UPSERT_ERROR_MESSAGE = "Failed to upsert embeddings into pgvector."
_CONFIG_DISABLED_MESSAGE = "pgvector ingestion is disabled by configuration"
_MISSING_TABLE_MESSAGE = "PGVECTOR_TABLE_NAME is required for ingestion"
_MISSING_SCHEMA_MESSAGE = "POSTGRES_SCHEMA is required for ingestion"


class IngestError(Exception):
    """Raised when ingestion fails."""


class ChunkerStrategy(StrEnum):
    """Available chunking strategies."""

    SLIDING_WINDOW = "sliding_window"
    PARAGRAPH = "paragraph"


class TextDocument(BaseModel):
    """A text document loaded from disk."""

    path: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """Chunked representation of a document ready for embedding."""

    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SupportsCursor(Protocol):
    """Cursor interface for database operations."""

    def execute(self, query: object, params: Sequence[object]) -> object:
        """Execute a parametrized SQL query."""
        ...

    def __enter__(self) -> Self:  # pragma: no cover - protocol
        """Enter the context manager."""
        ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> object:  # pragma: no cover - protocol
        """Exit the context manager."""
        ...


class SupportsTransactionalConnection(Protocol):
    """Connection interface exposing transactional controls."""

    def cursor(self) -> SupportsCursor:
        """Return a cursor object."""
        ...

    def commit(self) -> None:
        """Commit the current transaction."""
        ...

    def rollback(self) -> None:
        """Rollback the current transaction."""
        ...

    def __enter__(self) -> Self:  # pragma: no cover - protocol
        """Enter the context manager."""
        ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> object:  # pragma: no cover - protocol
        """Exit the context manager."""
        ...


class ConnectionPoolProtocol(Protocol):
    """Connection pool interface compatible with psycopg pools."""

    def connection(self, timeout: float | None = None) -> SupportsTransactionalConnection:
        """Acquire a managed connection from the pool."""
        ...


class BaseChunker(ABC):
    """Abstract chunking strategy."""

    @abstractmethod
    def chunk(self, document: TextDocument) -> list[DocumentChunk]:
        """Split a document into chunks."""


class SlidingWindowChunker(BaseChunker):
    """Chunker using a fixed window with overlap."""

    def __init__(self, *, chunk_size: int = 500, overlap: int = 50) -> None:
        """Configure the chunk size and overlap."""
        if chunk_size <= 0:
            msg = "chunk_size must be positive"
            raise ValueError(msg)
        if overlap < 0 or overlap >= chunk_size:
            msg = "overlap must be non-negative and smaller than chunk_size"
            raise ValueError(msg)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: TextDocument) -> list[DocumentChunk]:
        """Split the text into fixed-size overlapping chunks."""
        text = document.content.strip()
        if not text:
            return []

        chunks: list[DocumentChunk] = []
        step = self.chunk_size - self.overlap
        index = 0
        for start in range(0, len(text), step):
            chunk_text = text[start : start + self.chunk_size]
            if not chunk_text:
                continue
            chunk_id = f"{document.path}::chunk-{index}"
            chunk_metadata = {
                **document.metadata,
                "chunk_index": index,
                "chunk_start": start,
                "chunk_end": start + len(chunk_text),
            }
            chunks.append(
                DocumentChunk(id=chunk_id, content=chunk_text, metadata=chunk_metadata),
            )
            index += 1
        return chunks


class ParagraphChunker(BaseChunker):
    """Chunker that splits documents by paragraph blocks."""

    def __init__(self, *, min_length: int = 1) -> None:
        """Configure the minimal paragraph length to keep."""
        if min_length < 1:
            msg = "min_length must be at least 1"
            raise ValueError(msg)
        self.min_length = min_length

    def chunk(self, document: TextDocument) -> list[DocumentChunk]:
        """Split the document into paragraph chunks."""
        paragraphs = [segment.strip() for segment in document.content.split("\n\n")]
        chunks: list[DocumentChunk] = []
        for index, paragraph in enumerate(paragraphs):
            if len(paragraph) < self.min_length:
                continue
            chunk_id = f"{document.path}::paragraph-{index}"
            chunk_metadata = {**document.metadata, "chunk_index": index}
            chunks.append(
                DocumentChunk(id=chunk_id, content=paragraph, metadata=chunk_metadata),
            )
        return chunks


def build_chunker(
    strategy: ChunkerStrategy | str,
    *,
    registry: Mapping[str, type[BaseChunker]] | None = None,
    **kwargs: Any,
) -> BaseChunker:
    """Instantiate a chunker based on the chosen strategy."""
    normalized_registry: dict[str, type[BaseChunker]] = {
        ChunkerStrategy.SLIDING_WINDOW.value: SlidingWindowChunker,
        ChunkerStrategy.PARAGRAPH.value: ParagraphChunker,
        **dict(registry or {}),
    }
    key = strategy.value if isinstance(strategy, ChunkerStrategy) else str(strategy)
    try:
        chunker_cls = normalized_registry[key]
    except KeyError as exc:
        msg = f"Unsupported chunker strategy: {strategy}"
        raise IngestError(msg) from exc
    return chunker_cls(**kwargs)


def load_text_documents(directory: Path) -> list[TextDocument]:
    """Recursively load ``.txt`` files under the given directory."""
    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        msg = f"Directory not found: {directory_path}"
        raise IngestError(msg)

    documents: list[TextDocument] = []
    for path in sorted(directory_path.rglob("*.txt")):
        content = path.read_text(encoding="utf-8")
        metadata = {"path": str(path), "name": path.name, "parent": str(path.parent)}
        documents.append(
            TextDocument(path=str(path), content=content, metadata=metadata),
        )
    return documents


def chunk_documents(documents: Sequence[TextDocument], chunker: BaseChunker) -> list[DocumentChunk]:
    """Chunk all documents using the provided chunker."""
    chunks: list[DocumentChunk] = []
    for document in documents:
        chunks.extend(chunker.chunk(document))
    return chunks


def _table_identifier(schema: str, table: str) -> sql.Identifier:
    return sql.Identifier(schema, table)


def _try_import_execute_values() -> (
    Callable[
        [SupportsCursor, object, Sequence[tuple[object, ...]], str | None],
        object,
    ]
    | None
):
    try:
        module = importlib.import_module("psycopg.extras")
    except ModuleNotFoundError:
        return None
    execute_values_fn = getattr(module, "execute_values", None)
    if execute_values_fn is None:
        return None
    return cast(
        "Callable[[SupportsCursor, object, Sequence[tuple[object, ...]], str | None], object]",
        execute_values_fn,
    )


execute_values = _try_import_execute_values()


def _upsert_embeddings(
    pool: ConnectionPoolProtocol,
    rows: Sequence[tuple[DocumentChunk, Sequence[float]]],
    *,
    schema: str,
    table: str,
    max_retries: int = 3,
    timeout: float = 5.0,
    sleep_base: float = 0.5,
) -> None:
    if not rows:
        return

    table_reference = _table_identifier(schema, table)
    batched_query = sql.SQL(
        "INSERT INTO {table} (id, content, metadata, embedding) VALUES %s "
        "ON CONFLICT (id) DO UPDATE SET "
        "content = EXCLUDED.content, "
        "metadata = EXCLUDED.metadata, "
        "embedding = EXCLUDED.embedding",
    ).format(table=table_reference)
    values_template = "(%s, %s, %s::jsonb, %s::vector)"

    values = [(chunk.id, chunk.content, json.dumps(chunk.metadata), list(embedding)) for chunk, embedding in rows]

    backoff = sleep_base
    last_error: Exception | None = None
    execute_values_fn = execute_values or _try_import_execute_values()
    for attempt in range(max_retries + 1):
        connection: SupportsTransactionalConnection | None = None
        try:
            with pool.connection(timeout=timeout) as connection:
                with connection.cursor() as cursor:
                    if execute_values_fn is None:
                        placeholders = sql.SQL(", ").join(sql.SQL(values_template) for _ in values)
                        combined_query = sql.SQL(
                            "INSERT INTO {table} (id, content, metadata, embedding) VALUES {values} "
                            "ON CONFLICT (id) DO UPDATE SET "
                            "content = EXCLUDED.content, "
                            "metadata = EXCLUDED.metadata, "
                            "embedding = EXCLUDED.embedding",
                        ).format(table=table_reference, values=placeholders)
                        parameters: list[object] = []
                        for value in values:
                            parameters.extend(value)
                        cursor.execute(combined_query, parameters)
                    else:
                        execute_values_fn(
                            cursor,
                            batched_query,
                            values,
                            values_template,
                        )
                connection.commit()
        except PsycopgError as exc:
            last_error = exc
            if connection is not None:
                connection.rollback()
            if attempt >= max_retries:
                raise IngestError(_UPSERT_ERROR_MESSAGE) from exc
            time.sleep(backoff)
            backoff *= 2
        else:
            return
    if last_error is not None:
        raise IngestError(_UPSERT_ERROR_MESSAGE) from last_error


def ingest_text_directory(
    directory: Path,
    *,
    pool: ConnectionPoolProtocol,
    chunker: BaseChunker | None = None,
    chunk_strategy: ChunkerStrategy | str = ChunkerStrategy.SLIDING_WINDOW,
    chunker_registry: Mapping[str, type[BaseChunker]] | None = None,
    chunker_options: dict[str, Any] | None = None,
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI,
    embedding_model: str | None = None,
    batch_size: int = 10,
    max_retries: int = 3,
    timeout: float = 5.0,
    sleep_base: float = 0.5,
    embedding_fn: Callable[[DocumentChunk], Sequence[float]] | None = None,
    openai_client: SupportsOpenAIClient | None = None,
    google_client: SupportsGoogleClient | None = None,
    voyage_client: SupportsVoyageClient | None = None,
) -> None:
    """Load, chunk, embed, and upsert text files into pgvector."""
    if not settings.enable_pgvector:
        raise IngestError(_CONFIG_DISABLED_MESSAGE)
    if not settings.pgvector_table_name:
        raise IngestError(_MISSING_TABLE_MESSAGE)
    if not settings.postgres_schema:
        raise IngestError(_MISSING_SCHEMA_MESSAGE)

    chunker_instance = chunker or build_chunker(
        chunk_strategy,
        registry=chunker_registry,
        **(chunker_options or {}),
    )
    documents = load_text_documents(directory)
    chunks = chunk_documents(documents, chunker_instance)

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings: list[tuple[DocumentChunk, Sequence[float]]] = []
        if embedding_fn is not None:
            embeddings.extend((chunk, embedding_fn(chunk)) for chunk in batch)
        else:
            vectors = generate_embeddings_batch(
                [chunk.content for chunk in batch],
                provider=embedding_provider,
                model=embedding_model,
                timeout=timeout,
                max_retries=max_retries,
                openai_client=openai_client,
                google_client=google_client,
                voyage_client=voyage_client,
            )
            embeddings.extend(zip(batch, vectors, strict=True))

        _upsert_embeddings(
            pool,
            embeddings,
            schema=settings.postgres_schema,
            table=settings.pgvector_table_name,
            max_retries=max_retries,
            timeout=timeout,
            sleep_base=sleep_base,
        )
