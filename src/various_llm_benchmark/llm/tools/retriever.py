"""Utilities for database-backed retrieval with embeddings and full-text search."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import StrEnum
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, Self, cast

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from openai.types import CreateEmbeddingResponse
    from psycopg import sql

from various_llm_benchmark.settings import settings

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from openai.types import CreateEmbeddingResponse


class RetrieverError(Exception):
    """Base exception for retriever failures."""


class EmbeddingProvider(StrEnum):
    """Supported embedding providers."""

    OPENAI = "openai"
    GOOGLE = "google"
    VOYAGE = "voyage"


class RetrievedDocument(BaseModel):
    """Normalized representation of a retrieved document."""

    id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    score: float = Field(ge=0)
    metadata: dict[str, object] = Field(default_factory=dict)
    source: str = Field(min_length=1)


def _empty_retrieved_documents() -> list[RetrievedDocument]:
    return []


class RetrieverResponse(BaseModel):
    """Container for retriever search results."""

    documents: list[RetrievedDocument] = Field(default_factory=_empty_retrieved_documents)


class SupportsEmbeddingCreate(Protocol):
    """Minimal interface for an embeddings client."""

    def create(self, **kwargs: object) -> object:
        """Create an embedding."""
        ...


class SupportsOpenAIClient(Protocol):
    """Client interface exposing an embeddings attribute."""

    embeddings: SupportsEmbeddingCreate


class SupportsGoogleModels(Protocol):
    """Google client interface exposing an embed_content method."""

    def embed_content(self, *, model: str, contents: list[str]) -> object:
        """Create an embedding from the provided contents."""
        ...


class SupportsGoogleClient(Protocol):
    """Client interface exposing models with embedding support."""

    models: SupportsGoogleModels


class SupportsVoyageClient(Protocol):
    """Client interface exposing an embed method."""

    def embed(self, *, texts: list[str], model: str, **kwargs: object) -> object:
        """Create an embedding from the provided texts."""
        ...


class SupportsCursor(Protocol):
    """Cursor interface used for database access."""

    def execute(self, query: object, params: Sequence[object]) -> object:
        """Execute a parametrized SQL query."""
        ...

    def fetchall(self) -> Sequence[tuple[object, ...]]:
        """Fetch all rows from the last query."""
        ...

    def __enter__(self) -> Self:
        """Enter the context manager."""
        ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> object:
        """Exit the context manager."""
        ...


class SupportsConnection(Protocol):
    """Connection interface exposing a cursor factory."""

    def cursor(self) -> SupportsCursor:
        """Return a cursor object."""
        ...

    def __enter__(self) -> Self:
        """Enter the context manager."""
        ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> object:
        """Exit the context manager."""
        ...


class ConnectionPoolProtocol(Protocol):
    """Connection pool interface compatible with psycopg pools."""

    def connection(self, timeout: float | None = None) -> AbstractContextManager[SupportsConnection]:
        """Acquire a managed connection from the pool."""
        ...


def create_postgres_pool(
    *,
    conninfo: str | None = None,
    min_size: int = 1,
    max_size: int = 5,
    timeout: float = 5.0,
    max_lifetime: float = 60.0,
) -> ConnectionPoolProtocol:
    """Create a PostgreSQL connection pool with sensible defaults."""
    connection_string = conninfo or settings.postgres_connection_string.get_secret_value()
    if not connection_string:
        msg = "Connection string is required to build a PostgreSQL pool."
        raise RetrieverError(msg)

    try:
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(
            conninfo=connection_string,
            min_size=min_size,
            max_size=max_size,
            timeout=timeout,
            max_lifetime=max_lifetime,
        )
    except Exception as exc:  # pragma: no cover - psycopg raises specific subclasses
        msg = "Failed to create PostgreSQL connection pool."
        raise RetrieverError(msg) from exc

    return cast("ConnectionPoolProtocol", pool)


def generate_embedding(
    text: str,
    *,
    provider: EmbeddingProvider,
    model: str | None = None,
    timeout: float = 30.0,
    max_retries: int = 2,
    openai_client: SupportsOpenAIClient | None = None,
    google_client: SupportsGoogleClient | None = None,
    voyage_client: SupportsVoyageClient | None = None,
) -> list[float]:
    """Generate an embedding vector using the requested provider with retries."""
    if provider is EmbeddingProvider.OPENAI:
        provider_handler = partial(
            _generate_openai_embedding,
            text,
            model=model,
            timeout=timeout,
            client=openai_client,
        )
    elif provider is EmbeddingProvider.GOOGLE:
        provider_handler = partial(
            _generate_google_embedding,
            text,
            model=model,
            timeout=timeout,
            client=google_client,
        )
    elif provider is EmbeddingProvider.VOYAGE:
        provider_handler = partial(
            _generate_voyage_embedding,
            text,
            model=model,
            timeout=timeout,
            client=voyage_client,
        )
    else:
        msg = f"Unsupported embedding provider: {provider}"
        raise RetrieverError(msg)

    backoff: float = 0.2
    errors: list[Exception] = []
    for attempt in range(max_retries + 1):
        try:
            return provider_handler()
        except Exception as exc:
            from openai import APIConnectionError, APITimeoutError, RateLimitError

            if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
                errors.append(exc)
                if attempt >= max_retries:
                    msg = "Embedding request failed after retries."
                    raise RetrieverError(msg) from exc
                time.sleep(backoff)
                backoff *= 2
                continue
            # defensive guard
            errors.append(exc)
            if attempt >= max_retries:
                msg = "Embedding request failed unexpectedly."
                raise RetrieverError(msg) from exc
            time.sleep(backoff)
            backoff *= 2

    if errors:
        last_error = errors[-1]
        msg = "Embedding generation exhausted retries."
        raise RetrieverError(msg) from last_error

    msg = "Embedding generation failed without raising errors."
    raise RetrieverError(msg)


def generate_embeddings_batch(
    texts: list[str],
    *,
    provider: EmbeddingProvider,
    model: str | None = None,
    timeout: float = 30.0,
    max_retries: int = 2,
    openai_client: SupportsOpenAIClient | None = None,
    google_client: SupportsGoogleClient | None = None,
    voyage_client: SupportsVoyageClient | None = None,
) -> list[list[float]]:
    """Generate embeddings for multiple texts in a single request when possible."""
    if not texts:
        return []

    handler = _select_batch_handler(
        texts,
        provider=provider,
        model=model,
        timeout=timeout,
        max_retries=max_retries,
        openai_client=openai_client,
        google_client=google_client,
        voyage_client=voyage_client,
    )

    backoff: float = 0.2
    errors: list[Exception] = []
    for attempt in range(max_retries + 1):
        try:
            return handler()
        except Exception as exc:
            from openai import APIConnectionError, APITimeoutError, RateLimitError

            if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
                errors.append(exc)
                if attempt >= max_retries:
                    msg = "Batch embedding request failed after retries."
                    raise RetrieverError(msg) from exc
                time.sleep(backoff)
                backoff *= 2
                continue
            # defensive guard
            errors.append(exc)
            if attempt >= max_retries:
                msg = "Batch embedding request failed unexpectedly."
                raise RetrieverError(msg) from exc
            time.sleep(backoff)
            backoff *= 2

    if errors:
        last_error = errors[-1]
        msg = "Batch embedding generation exhausted retries."
        raise RetrieverError(msg) from last_error

    msg = "Batch embedding generation failed without raising errors."
    raise RetrieverError(msg)


def pgvector_similarity_search(
    pool: ConnectionPoolProtocol,
    embedding: Sequence[float],
    *,
    schema: str,
    table: str,
    top_k: int | None = None,
    threshold: float | None = None,
    timeout: float = 5.0,
) -> list[RetrievedDocument]:
    """Search pgvector index using the provided embedding."""
    table_reference = _qualified_table(schema, table)
    similarity_threshold = threshold if threshold is not None else settings.search_score_threshold
    limit = top_k or settings.search_top_k
    from psycopg import sql

    query = sql.SQL(
        "SELECT id, content, metadata, 1 - (embedding <=> %s::vector) AS score "
        "FROM {table} "
        "WHERE 1 - (embedding <=> %s::vector) >= %s "
        "ORDER BY score DESC LIMIT %s",
    ).format(table=table_reference)
    params: list[object] = [list(embedding), list(embedding), similarity_threshold, limit]

    try:
        with pool.connection(timeout=timeout) as connection, connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cast("Sequence[RowRecord]", cursor.fetchall())
    except Exception as exc:
        from psycopg import Error as PsycopgError
        from psycopg_pool import PoolTimeout

        if isinstance(exc, (PoolTimeout, PsycopgError)):
            msg = "pgvector similarity search failed."
            raise RetrieverError(msg) from exc
        raise

    documents: list[RetrievedDocument] = []
    for row in rows:
        if len(row) < _MIN_RESULT_COLUMNS:
            continue
        score = float(row[3])
        if score < similarity_threshold:
            continue
        documents.append(
            RetrievedDocument(
                id=str(row[0]),
                content=str(row[1]),
                metadata=_coerce_metadata(row[2]),
                score=score,
                source="pgvector",
            ),
        )
    return documents


def pgroonga_full_text_search(
    pool: ConnectionPoolProtocol,
    query_text: str,
    *,
    schema: str,
    table: str,
    top_k: int | None = None,
    threshold: float | None = None,
    timeout: float = 5.0,
) -> list[RetrievedDocument]:
    """Run a PGroonga full-text search query."""
    table_reference = _qualified_table(schema, table)
    rank_threshold = threshold if threshold is not None else settings.search_score_threshold
    limit = top_k or settings.search_top_k
    from psycopg import sql

    query = sql.SQL(
        "SELECT id, content, metadata, pgroonga_score(tableoid, ctid) AS score "
        "FROM {table} "
        "WHERE content &@ %s "
        "ORDER BY score DESC LIMIT %s",
    ).format(table=table_reference)
    params: list[object] = [query_text, limit]

    try:
        with pool.connection(timeout=timeout) as connection, connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cast("Sequence[RowRecord]", cursor.fetchall())
    except Exception as exc:
        from psycopg import Error as PsycopgError
        from psycopg_pool import PoolTimeout

        if isinstance(exc, (PoolTimeout, PsycopgError)):
            msg = "PGroonga full-text search failed."
            raise RetrieverError(msg) from exc
        raise

    documents: list[RetrievedDocument] = []
    for row in rows:
        if len(row) < _MIN_RESULT_COLUMNS:
            continue
        score = float(row[3])
        if score < rank_threshold:
            continue
        documents.append(
            RetrievedDocument(
                id=str(row[0]),
                content=str(row[1]),
                metadata=_coerce_metadata(row[2]),
                score=score,
                source="pgroonga",
            ),
        )
    return documents


def merge_ranked_results(
    vector_results: Sequence[RetrievedDocument],
    text_results: Sequence[RetrievedDocument],
    *,
    vector_weight: float = 0.6,
    text_weight: float = 0.4,
) -> list[RetrievedDocument]:
    """Normalize scores per source and merge them with weights."""
    normalized_vectors = _normalize_scores(vector_results)
    normalized_texts = _normalize_scores(text_results)

    aggregated: dict[str, RetrievedDocument] = {}
    scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}

    for result, weight in ((normalized_vectors, vector_weight), (normalized_texts, text_weight)):
        for item in result:
            current = aggregated.get(item.id)
            if current is None:
                aggregated[item.id] = item
                scores[item.id] = 0.0
                sources[item.id] = {item.source}
            else:
                merged_metadata = {**current.metadata, **item.metadata}
                aggregated[item.id] = current.model_copy(update={"metadata": merged_metadata})
                sources[item.id].add(item.source)
            scores[item.id] += item.score * weight

    merged_results: list[RetrievedDocument] = []
    for doc_id, base in aggregated.items():
        metadata = dict(base.metadata)
        metadata["sources"] = sorted(sources.get(doc_id, {base.source}))
        merged_results.append(
            base.model_copy(update={"score": scores[doc_id], "metadata": metadata}),
        )

    merged_results.sort(key=lambda item: item.score, reverse=True)
    return merged_results


def _generate_openai_embedding(
    text: str,
    *,
    model: str | None,
    timeout: float,
    client: SupportsOpenAIClient | None,
) -> list[float]:
    embedding_model = model or settings.embedding_model or settings.openai_embedding_model
    openai_client = cast(
        "SupportsOpenAIClient",
        client
        or (
            (  # noqa: PLC3002 - lazy import pattern
                lambda: (
                    __import__("openai").OpenAI(
                        api_key=settings.openai_api_key.get_secret_value(),
                        timeout=timeout,
                    )
                )
            )()
        ),
    )
    response = cast(
        "CreateEmbeddingResponse",
        openai_client.embeddings.create(
            model=embedding_model,
            input=[text],
            timeout=timeout,
        ),
    )
    return _extract_embedding_vector(response)


def _generate_openai_embeddings_batch(
    texts: Sequence[str],
    *,
    model: str | None,
    timeout: float,
    client: SupportsOpenAIClient | None,
) -> list[list[float]]:
    embedding_model = model or settings.embedding_model or settings.openai_embedding_model
    openai_client = cast(
        "SupportsOpenAIClient",
        client
        or (
            (  # noqa: PLC3002 - lazy import pattern
                lambda: (
                    __import__("openai").OpenAI(
                        api_key=settings.openai_api_key.get_secret_value(),
                        timeout=timeout,
                    )
                )
            )()
        ),
    )
    response = cast(
        "CreateEmbeddingResponse",
        openai_client.embeddings.create(
            model=embedding_model,
            input=list(texts),
            timeout=timeout,
        ),
    )
    return _extract_embedding_vectors(response, expected=len(texts))


def _generate_google_embedding(
    text: str,
    *,
    model: str | None,
    timeout: float,
    client: SupportsGoogleClient | None,
) -> list[float]:
    embedding_model = model or settings.embedding_model or settings.google_embedding_model
    from google.genai import Client
    from google.genai.types import HttpOptions

    http_options = HttpOptions(timeout=max(0, int(timeout)))
    genai_client = client or Client(
        api_key=settings.gemini_api_key.get_secret_value(),
        http_options=http_options,
    )
    response = genai_client.models.embed_content(
        model=embedding_model,
        contents=[text],
    )
    return _extract_embedding_vector(response)


def _generate_google_embeddings_batch(
    texts: Sequence[str],
    *,
    model: str | None,
    timeout: float,
    client: SupportsGoogleClient | None,
    max_retries: int,
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    backoff: float = 0.1
    for index, text in enumerate(texts):
        current_retries = 0
        while True:
            try:
                embeddings.append(
                    _generate_google_embedding(
                        text,
                        model=model,
                        timeout=timeout,
                        client=client,
                    ),
                )
                break
            except Exception as exc:
                from openai import APIConnectionError, APITimeoutError, RateLimitError

                if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
                    if current_retries >= max_retries:
                        msg = f"Embedding request failed for item {index}"
                        raise RetrieverError(msg) from exc
                    time.sleep(backoff)
                    backoff *= 2
                    current_retries += 1
                    continue
                raise
    return embeddings


def _generate_voyage_embedding(
    text: str,
    *,
    model: str | None,
    timeout: float,
    client: SupportsVoyageClient | None,
) -> list[float]:
    embedding_model = model or settings.embedding_model or settings.voyage_embedding_model
    if client:
        voyage_client = client
    else:
        from voyageai.client import Client as VoyageClient

        voyage_client = VoyageClient(api_key=settings.voyage_api_key.get_secret_value(), timeout=timeout)
    response = voyage_client.embed(
        texts=[text],
        model=embedding_model,
    )
    return _extract_embedding_vector(response)


def _generate_voyage_embeddings_batch(
    texts: Sequence[str],
    *,
    model: str | None,
    timeout: float,
    client: SupportsVoyageClient | None,
) -> list[list[float]]:
    embedding_model = model or settings.embedding_model or settings.voyage_embedding_model
    if client:
        voyage_client = client
    else:
        from voyageai.client import Client as VoyageClient

        voyage_client = VoyageClient(api_key=settings.voyage_api_key.get_secret_value(), timeout=timeout)
    response = voyage_client.embed(
        texts=list(texts),
        model=embedding_model,
    )
    return _extract_embedding_vectors(response, expected=len(texts))


def _extract_embedding_vector(payload: object) -> list[float]:
    payload_obj: Any = payload
    if isinstance(payload_obj, Sequence) and not isinstance(payload_obj, (str, bytes)):
        sequence_payload: Sequence[float] = cast("Sequence[float]", payload_obj)
        return _to_float_list(sequence_payload)

    embedding_attr: Any = getattr(payload_obj, "embedding", None)
    if embedding_attr is not None:
        candidate = getattr(embedding_attr, "values", embedding_attr)
        return _to_float_list(_ensure_iterable(cast("Iterable[float]", candidate)))

    data_attr: Any = getattr(payload_obj, "data", None)
    if isinstance(data_attr, Sequence) and data_attr:
        data_sequence: Sequence[Any] = cast("Sequence[Any]", data_attr)
        first: Any = data_sequence[0]
        nested_embedding: Any = getattr(first, "embedding", None)
        if nested_embedding is not None:
            return _to_float_list(_ensure_iterable(cast("Iterable[float]", nested_embedding)))
        if isinstance(first, Mapping) and "embedding" in first:
            return _to_float_list(_ensure_iterable(cast("Iterable[float]", first["embedding"])))

    embeddings_attr: Any = getattr(payload_obj, "embeddings", None)
    if isinstance(embeddings_attr, Sequence) and embeddings_attr:
        embeddings_sequence: Sequence[Any] = cast("Sequence[Any]", embeddings_attr)
        first_embedding: Any = getattr(embeddings_sequence[0], "values", embeddings_sequence[0])
        return _to_float_list(_ensure_iterable(cast("Iterable[float]", first_embedding)))

    if isinstance(payload_obj, Mapping):
        mapping_payload: Mapping[str, object] = cast("Mapping[str, object]", payload_obj)
        candidate: Any = mapping_payload.get("embedding") or mapping_payload.get("vector")
        if candidate is not None:
            return _to_float_list(_ensure_iterable(cast("Iterable[float]", candidate)))

    msg = "Embedding payload does not contain an embedding vector."
    raise RetrieverError(msg)


def _extract_embedding_vectors(payload: object, *, expected: int) -> list[list[float]]:
    payload_obj: object = payload
    if isinstance(payload_obj, (str, bytes)):
        msg = "Embedding payload does not contain batch embeddings."
        raise RetrieverError(msg)

    vectors = _extract_from_sequence_payload(payload_obj)
    if vectors is None:
        vectors = _extract_from_data_attr(payload_obj)
    if vectors is None:
        vectors = _extract_from_embeddings_attr(payload_obj)

    if vectors is None:
        msg = "Embedding payload does not contain batch embeddings."
        raise RetrieverError(msg)

    if expected and len(vectors) != expected:
        msg = f"Expected {expected} embeddings but received {len(vectors)}"
        raise RetrieverError(msg)

    return vectors


def _extract_from_sequence_payload(payload_obj: object) -> list[list[float]] | None:
    if isinstance(payload_obj, Sequence) and payload_obj and not isinstance(payload_obj, (str, bytes)):
        sequence_payload: Sequence[Any] = cast("Sequence[Any]", payload_obj)
        return [_to_float_list(_ensure_iterable(cast("Iterable[float]", item))) for item in sequence_payload]
    return None


def _extract_from_data_attr(payload_obj: object) -> list[list[float]] | None:
    payload_any = cast("Any", payload_obj)
    data_attr: Any = getattr(payload_any, "data", None)
    if not isinstance(data_attr, Sequence):
        return None

    data_sequence: Sequence[Any] = cast("Sequence[Any]", data_attr)
    vectors: list[list[float]] = []
    for item in data_sequence:
        embedding_attr = cast("Iterable[float] | None", getattr(item, "embedding", None))
        if embedding_attr is None and isinstance(item, Mapping):
            mapping_item: Mapping[str, object] = cast("Mapping[str, object]", item)
            embedding_attr = cast("Iterable[float] | None", mapping_item.get("embedding"))
        if embedding_attr is None:
            msg = "Embedding payload does not contain an embedding vector."
            raise RetrieverError(msg)
        vectors.append(_to_float_list(_ensure_iterable(embedding_attr)))
    return vectors


def _extract_from_embeddings_attr(payload_obj: object) -> list[list[float]] | None:
    payload_any = cast("Any", payload_obj)
    embeddings_attr: Any = getattr(payload_any, "embeddings", None)
    if not (isinstance(embeddings_attr, Sequence) and embeddings_attr):
        return None

    embeddings_sequence: Sequence[Any] = cast("Sequence[Any]", embeddings_attr)
    return [
        _to_float_list(_ensure_iterable(cast("Iterable[float]", getattr(item, "values", item))))
        for item in embeddings_sequence
    ]


def _select_batch_handler(
    texts: Sequence[str],
    *,
    provider: EmbeddingProvider,
    model: str | None,
    timeout: float,
    max_retries: int,
    openai_client: SupportsOpenAIClient | None,
    google_client: SupportsGoogleClient | None,
    voyage_client: SupportsVoyageClient | None,
) -> Callable[[], list[list[float]]]:
    if provider is EmbeddingProvider.OPENAI:
        return partial(
            _generate_openai_embeddings_batch,
            texts,
            model=model,
            timeout=timeout,
            client=openai_client,
        )
    if provider is EmbeddingProvider.GOOGLE:
        return partial(
            _generate_google_embeddings_batch,
            texts,
            model=model,
            timeout=timeout,
            client=google_client,
            max_retries=max_retries,
        )
    if provider is EmbeddingProvider.VOYAGE:
        return partial(
            _generate_voyage_embeddings_batch,
            texts,
            model=model,
            timeout=timeout,
            client=voyage_client,
        )

    msg = f"Unsupported embedding provider: {provider}"
    raise RetrieverError(msg)


def _normalize_scores(results: Sequence[RetrievedDocument]) -> list[RetrievedDocument]:
    if not results:
        return []

    scores = [item.score for item in results]
    max_score = max(scores)
    min_score = min(scores)
    if math.isclose(max_score, min_score):
        normalized = [1.0 for _ in results]
    else:
        normalized = [(score - min_score) / (max_score - min_score) for score in scores]

    return [
        item.model_copy(update={"score": normalized_score})
        for item, normalized_score in zip(results, normalized, strict=True)
    ]


def _to_float_list(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def _qualified_table(schema: str, table: str) -> sql.Identifier:
    _validate_identifier(table, "table")
    from psycopg import sql

    if schema:
        _validate_identifier(schema, "schema")
        return sql.Identifier(schema, table)
    return sql.Identifier(table)


def _validate_identifier(value: str, label: str) -> None:
    if not value or not value.replace("_", "a").isalnum() or not (value[0].isalpha() or value[0] == "_"):
        msg = f"Invalid {label} name: {value!r}"
        raise RetrieverError(msg)


RowRecord = tuple[object, object, Mapping[str, object] | None, float | int]
_MIN_RESULT_COLUMNS = 4


def _coerce_metadata(value: Any) -> dict[str, object]:
    if isinstance(value, Mapping):
        mapping_value: Mapping[str, object] = cast("Mapping[str, object]", value)
        return {str(key): item for key, item in mapping_value.items()}
    if value is None:
        return {}
    return {"raw": value}


def _ensure_iterable(value: Any) -> Iterable[float]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return cast("Iterable[float]", value)
    msg = "Embedding value must be an iterable of floats."
    raise RetrieverError(msg)


__all__ = [
    "EmbeddingProvider",
    "RetrievedDocument",
    "RetrieverError",
    "RetrieverResponse",
    "create_postgres_pool",
    "generate_embedding",
    "generate_embeddings_batch",
    "merge_ranked_results",
    "pgroonga_full_text_search",
    "pgvector_similarity_search",
]
