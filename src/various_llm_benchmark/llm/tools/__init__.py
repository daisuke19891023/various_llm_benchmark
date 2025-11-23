"""Utilities for invoking model-native tools."""

from importlib import import_module

from various_llm_benchmark.llm.tools.payloads import (
    to_agents_sdk_tools_payload,
    to_agno_tools_payload,
    to_anthropic_tools_payload,
    to_gemini_tools_payload,
    to_google_adk_tools_payload,
    to_openai_tools_payload,
)
from various_llm_benchmark.llm.tools.retriever import (
    EmbeddingProvider,
    RetrievedDocument,
    RetrieverError,
    create_postgres_pool,
    generate_embedding,
    merge_ranked_results,
    pgroonga_full_text_search,
    pgvector_similarity_search,
)
from various_llm_benchmark.llm.tools.selector import ToolSelector


def _register_builtin_tools() -> None:
    """Eagerly import builtin tool modules to populate the registry."""
    for module_path in (
        "various_llm_benchmark.llm.tools.code_execution",
        "various_llm_benchmark.llm.tools.shell_execution",
        "various_llm_benchmark.llm.tools.builtin_memory",
    ):
        import_module(module_path)


_register_builtin_tools()

__all__ = [
    "EmbeddingProvider",
    "RetrievedDocument",
    "RetrieverError",
    "ToolSelector",
    "create_postgres_pool",
    "generate_embedding",
    "merge_ranked_results",
    "pgroonga_full_text_search",
    "pgvector_similarity_search",
    "to_agents_sdk_tools_payload",
    "to_agno_tools_payload",
    "to_anthropic_tools_payload",
    "to_gemini_tools_payload",
    "to_google_adk_tools_payload",
    "to_openai_tools_payload",
]
