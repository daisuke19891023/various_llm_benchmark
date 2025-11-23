"""Utilities for constructing database retriever tool callers."""

from __future__ import annotations

from functools import partial
from typing import Literal, Protocol, cast

from various_llm_benchmark.llm.tools import ToolSelector
from various_llm_benchmark.llm.tools.registry import (
    NativeToolType,
    RETRIEVER_INPUT_SCHEMA,
    RETRIEVER_TAG,
    RETRIEVER_TOOL_NAMESPACE,
    ToolCategory,
    ToolRegistration,
    register_tool,
)
from various_llm_benchmark.llm.tools.retriever import (
    EmbeddingProvider,
    RetrievedDocument,
    create_postgres_pool,
    generate_embedding,
    merge_ranked_results,
    pgroonga_full_text_search,
    pgvector_similarity_search,
)
from various_llm_benchmark.settings import settings

ProviderName = Literal["openai", "anthropic", "gemini"]


class RetrieverHandler(Protocol):
    """Callable that executes a retriever query."""

    def __call__(
        self,
        query: str,
        *,
        model: str | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
        timeout: float = 5.0,
    ) -> dict[str, object]:
        """Execute a retriever request."""
        ...


class RetrieverExecutor(Protocol):
    """Callable that runs a retriever query."""

    def __call__(
        self,
        query: str,
        *,
        model: str | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
        timeout: float = 5.0,
    ) -> dict[str, object]:
        """Execute a retriever request."""
        ...


def _tool_id(provider: ProviderName) -> str:
    return f"{RETRIEVER_TOOL_NAMESPACE}/{provider}"


def _serialize_document(document: RetrievedDocument) -> dict[str, object]:
    return {
        "id": document.id,
        "content": document.content,
        "metadata": document.metadata,
        "score": document.score,
        "source": document.source,
    }


def _register_retriever_tool(
    provider: ProviderName,
    description: str,
    handler: RetrieverHandler,
) -> None:
    register_tool(
        ToolRegistration(
            id=_tool_id(provider),
            name=f"{provider}-retriever",
            description=description,
            input_schema=RETRIEVER_INPUT_SCHEMA,
            tags={RETRIEVER_TAG, f"provider:{provider}"},
            native_type=NativeToolType.RETRIEVER,
            handler=handler,
            category=ToolCategory.BUILTIN,
        ),
    )


def _retrieve(
    provider: ProviderName,
    query: str,
    *,
    model: str | None = None,
    top_k: int | None = None,
    threshold: float | None = None,
    timeout: float = 5.0,
) -> dict[str, object]:
    pool = create_postgres_pool()
    embedding_provider = EmbeddingProvider(provider)
    vector_results = []
    if settings.enable_pgvector:
        embedding = generate_embedding(
            query,
            provider=embedding_provider,
            model=model,
            timeout=timeout,
        )
        vector_results = pgvector_similarity_search(
            pool,
            embedding,
            schema=settings.postgres_schema,
            table=settings.pgvector_table_name,
            top_k=top_k,
            threshold=threshold,
            timeout=timeout,
        )
    text_results = []
    if settings.enable_pgroonga:
        text_results = pgroonga_full_text_search(
            pool,
            query,
            schema=settings.postgres_schema,
            table=settings.pgroonga_table_name,
            top_k=top_k,
            threshold=threshold,
            timeout=timeout,
        )
    if vector_results and text_results:
        merged = merge_ranked_results(vector_results, text_results)
    elif vector_results:
        merged = vector_results
    else:
        merged = text_results

    return {"documents": [_serialize_document(doc) for doc in merged]}


def _ensure_retriever_tools_registered() -> None:
    """Register built-in retriever tool adapters."""
    for provider, description in (
        ("openai", "OpenAI 埋め込みでPGVector/PGroonga検索を行うビルトインツール"),
        ("anthropic", "Anthropic 埋め込みでPGVector/PGroonga検索を行うビルトインツール"),
        ("gemini", "Gemini 埋め込みでPGVector/PGroonga検索を行うビルトインツール"),
    ):
        try:
            _register_retriever_tool(
                cast("ProviderName", provider),
                description,
                cast("RetrieverHandler", partial(_retrieve, cast("ProviderName", provider))),
            )
        except ValueError:
            continue


def resolve_retriever_client(
    provider: ProviderName,
    *,
    category: ToolCategory = ToolCategory.BUILTIN,
) -> RetrieverExecutor:
    """Construct a retriever executor from the registry."""
    _ensure_retriever_tools_registered()
    selector = ToolSelector()
    registration = selector.select_one(
        category=category,
        names=[f"{provider}-retriever"],
        tags=[RETRIEVER_TAG, f"provider:{provider}"],
        ids=[_tool_id(provider)],
    )
    return cast("RetrieverExecutor", registration.handler)


__all__ = [
    "ProviderName",
    "resolve_retriever_client",
]
