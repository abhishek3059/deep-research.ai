"""Unit tests for the generation pipeline modules."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.generation import GenerationPipeline
from src.agents.llm_provider import LLMProvider
from src.agents.memory import ConversationMemory
from src.agents.prompts import FORMAT_INSTRUCTIONS, RESEARCH_PROMPT
from src.retrieval.pipeline import RetrievalMeta, RetrievalResult
from src.vectorstore.base import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_search_result(idx: int = 1, score: float = 0.9) -> SearchResult:
    """Build a minimal SearchResult for testing."""
    from src.ingestion.pipeline import ChunkMetadata

    meta = ChunkMetadata(
        source=f"doc_{idx}.pdf",
        source_type="pdf",
        page=idx,
        section=None,
        ingested_at="2025-01-01T00:00:00+00:00",
        content_hash="abc",
    )
    return SearchResult(id=f"chunk_{idx}", text=f"Chunk {idx} text", metadata=meta, score=score)


def _make_retrieval_result(n: int = 3) -> RetrievalResult:
    """Build a RetrievalResult with *n* search results."""
    return RetrievalResult(
        query="What is RAG?",
        expanded_queries=["What is RAG?", "retrieval augmented generation"],
        results=[_make_search_result(i) for i in range(1, n + 1)],
        retrieval_metadata=RetrievalMeta(
            strategy="hybrid",
            dense_results=n,
            sparse_results=0,
            reranked=False,
            latency_ms=42.0,
        ),
    )


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------

def test_system_prompt_formatting() -> None:
    """RESEARCH_PROMPT and FORMAT_INSTRUCTIONS contain required directives."""
    assert "ONLY" in RESEARCH_PROMPT
    assert "[Source N]" in RESEARCH_PROMPT
    assert "cannot" in RESEARCH_PROMPT.lower()

    assert "inline citations" in FORMAT_INSTRUCTIONS.lower()
    assert "summary" in FORMAT_INSTRUCTIONS.lower()


# ---------------------------------------------------------------------------
# Memory tests
# ---------------------------------------------------------------------------

def test_memory_add_and_get() -> None:
    """Messages added are returned in order via get_messages."""
    mem = ConversationMemory(window_size=5)
    mem.add_message("user", "Hello")
    mem.add_message("assistant", "Hi there")

    msgs = mem.get_messages()
    assert len(msgs) == 2
    assert msgs[0] == {"role": "user", "content": "Hello"}
    assert msgs[1] == {"role": "assistant", "content": "Hi there"}


def test_memory_window_limit() -> None:
    """Oldest messages are evicted when the window is exceeded."""
    mem = ConversationMemory(window_size=3)
    for i in range(5):
        mem.add_message("user", f"msg-{i}")

    msgs = mem.get_messages()
    assert len(msgs) == 3
    assert msgs[0]["content"] == "msg-2"
    assert msgs[1]["content"] == "msg-3"
    assert msgs[2]["content"] == "msg-4"


def test_memory_clear() -> None:
    """clear() empties the message list."""
    mem = ConversationMemory(window_size=5)
    mem.add_message("user", "test")
    mem.clear()
    assert mem.get_messages() == []


# ---------------------------------------------------------------------------
# GenerationPipeline tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_answer_returns_expected_keys() -> None:
    """generate_answer returns dict with answer, sources, query."""
    fake_llm = AsyncMock(spec=LLMProvider)
    fake_llm.generate = AsyncMock(return_value="RAG combines retrieval and generation.")

    pipeline = GenerationPipeline(llm_provider=fake_llm, memory=ConversationMemory())
    result = await pipeline.generate_answer("What is RAG?", _make_retrieval_result())

    assert "answer" in result
    assert "sources" in result
    assert "query" in result
    assert result["query"] == "What is RAG?"
    assert len(result["sources"]) == 3
    fake_llm.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_answer_context_includes_sources() -> None:
    """The context sent to the LLM contains [Source N] labels."""
    captured_messages: list[list[dict[str, str]]] = []

    async def _capture(messages: list[dict[str, str]], **_: object) -> str:
        captured_messages.append(messages)
        return "ok"

    fake_llm = AsyncMock(spec=LLMProvider)
    fake_llm.generate = _capture

    pipeline = GenerationPipeline(llm_provider=fake_llm, memory=ConversationMemory())
    await pipeline.generate_answer("test query", _make_retrieval_result(2))

    # The last user message should contain context with [Source 1] / [Source 2]
    last_user = [m for m in captured_messages[0] if m["role"] == "user"][-1]
    assert "[Source 1]" in last_user["content"]
    assert "[Source 2]" in last_user["content"]
    assert "[Source 3]" not in last_user["content"]
