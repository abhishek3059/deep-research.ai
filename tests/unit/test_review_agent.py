"""Unit tests for the Phase 2 self-check review agent and state machine.

All LLM and retrieval I/O is mocked.  These tests assert the self-check
contract: a weak answer is flagged by REVIEW, a revision improves it, the loop
guard bounds the REVIEW -> RESEARCH cycle, and a good answer passes untouched.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from src.agents.exceptions import AgentGraphError
from src.agents.generation import GenerationPipeline
from src.agents.graph_skeleton import ReviewGraph, build_review_graph
from src.agents.memory import ConversationMemory
from src.agents.review_agent import (
    REVIEW_MARKER,
    REVISION_MARKER,
    ReviewAgent,
)
from src.agents.state import REVIEW_DIMENSIONS, AgentState
from src.ingestion.pipeline import ChunkMetadata
from src.retrieval.pipeline import RetrievalMeta, RetrievalPipeline, RetrievalResult
from src.vectorstore.base import SearchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_search_result(idx: int = 1, score: float = 0.9) -> SearchResult:
    meta = ChunkMetadata(
        source=f"doc_{idx}.pdf",
        source_type="pdf",
        page=idx,
        section=None,
        ingested_at="2026-01-01T00:00:00+00:00",
        content_hash=f"hash_{idx}",
    )
    return SearchResult(
        id=f"chunk_{idx}", text=f"Context text {idx}", metadata=meta, score=score
    )


def _make_retrieval_result(query: str = "What is RAG?", n: int = 2) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        expanded_queries=[query, f"{query} expanded"],
        results=[_make_search_result(i) for i in range(1, n + 1)],
        retrieval_metadata=RetrievalMeta(
            strategy="hybrid",
            dense_results=n,
            sparse_results=n,
            reranked=True,
            latency_ms=12.0,
        ),
    )


def _review_json(passed: bool, failing: tuple[str, ...] = ()) -> str:
    """Build a well-formed review verdict JSON string."""
    dimensions = {
        name: {
            "passed": name not in failing,
            "issue": "" if name not in failing else f"bad {name}",
        }
        for name in REVIEW_DIMENSIONS
    }
    return json.dumps(
        {"passed": passed, "dimensions": dimensions, "summary": "verdict"}
    )


class _ScriptedLLM:
    """Fake LLM that routes calls by inspecting the system prompt marker."""

    def __init__(
        self,
        answers: list[str] | None = None,
        reviews: list[str] | None = None,
        revisions: list[str] | None = None,
    ) -> None:
        self._answers = answers or ["initial answer"]
        self._reviews = reviews or [_review_json(True)]
        self._revisions = revisions or ["revised answer"]
        self.generation_calls = 0
        self.review_calls = 0
        self.revision_calls = 0

    @staticmethod
    def _pick(sequence: list[str], index: int) -> str:
        return sequence[min(index, len(sequence) - 1)]

    async def generate(
        self, messages: list[dict[str, str]], temperature: float = 0.3
    ) -> str:
        system = messages[0]["content"]
        if REVIEW_MARKER in system:
            value = self._pick(self._reviews, self.review_calls)
            self.review_calls += 1
            return value
        if REVISION_MARKER in system:
            value = self._pick(self._revisions, self.revision_calls)
            self.revision_calls += 1
            return value
        value = self._pick(self._answers, self.generation_calls)
        self.generation_calls += 1
        return value


def _make_retrieval(results: list[RetrievalResult] | None = None) -> AsyncMock:
    retrieval = AsyncMock(spec=RetrievalPipeline)
    retrieval.retrieve = AsyncMock(
        side_effect=list(results or [_make_retrieval_result()])
    )
    return retrieval


def _make_agent(retrieval: AsyncMock, llm: _ScriptedLLM, **kwargs: object) -> ReviewAgent:
    generation = GenerationPipeline(
        llm_provider=llm, memory=ConversationMemory()  # type: ignore[arg-type]
    )
    return ReviewAgent(
        retrieval,
        generation_pipeline=generation,
        llm_provider=llm,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Review identifies issues in a weak answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_review_with_hallucinated_answer_flags_grounding_issue() -> None:
    """A hallucinated answer is flagged and routes back to research."""
    llm = _ScriptedLLM(
        reviews=[_review_json(False, failing=("grounding", "relevancy"))]
    )
    agent = _make_agent(_make_retrieval(), llm)
    state = AgentState(query="What is RAG?")
    state.initial_answer = "RAG was invented on Mars in 1987."
    state.add_research_results([_make_retrieval_result()])

    state = await agent.review(state)

    assert state.review_result is not None
    assert state.review_result.passed is False
    assert "grounding" in state.review_result.failed_dimensions
    assert "relevancy" in state.review_result.failed_dimensions
    assert "FAIL" in state.review_notes
    assert state.iteration_count == 1
    # Grounding is a context gap -> accumulate more evidence.
    assert state.status == "research"
    assert state.has_review is True


@pytest.mark.asyncio
async def test_review_with_incomplete_answer_routes_to_revise() -> None:
    """Relevancy/completeness failures route to REVISE, not re-retrieval."""
    llm = _ScriptedLLM(
        reviews=[_review_json(False, failing=("relevancy", "completeness"))]
    )
    agent = _make_agent(_make_retrieval(), llm)
    state = AgentState(query="What is RAG?")
    state.initial_answer = "a very incomplete answer"
    state.add_research_results([_make_retrieval_result()])

    state = await agent.review(state)

    assert state.status == "revise"


# ---------------------------------------------------------------------------
# Revision improves the answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_with_weak_answer_produces_revised_answer() -> None:
    """A failed first review triggers a revision that is delivered."""
    llm = _ScriptedLLM(
        answers=["weak answer"],
        reviews=[
            _review_json(False, failing=("relevancy", "completeness")),
            _review_json(True),
        ],
        revisions=["improved answer"],
    )
    agent = _make_agent(_make_retrieval(), llm)

    result = await agent.run("What is RAG?")

    assert result.initial_answer == "weak answer"
    assert result.revised is True
    assert result.answer == "improved answer"
    assert result.iterations == 2
    assert llm.revision_calls == 1
    assert result.review is not None
    assert result.review.passed is True


# ---------------------------------------------------------------------------
# Loop guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_with_persistent_failures_respects_loop_guard() -> None:
    """An answer that never passes stops at max_iterations, not forever."""
    llm = _ScriptedLLM(
        answers=["always weak"],
        reviews=[_review_json(False, failing=("relevancy", "completeness"))],
        revisions=["still weak"],
    )
    agent = _make_agent(_make_retrieval(), llm, max_iterations=3)

    result = await agent.run("What is RAG?")

    assert result.iterations == 3
    assert llm.review_calls == 3
    assert llm.revision_calls <= 3
    assert result.review is not None
    assert result.review.passed is False


@pytest.mark.asyncio
async def test_run_with_custom_max_iterations_bounds_reviews() -> None:
    """The guard honors an explicit max_iterations override."""
    llm = _ScriptedLLM(
        reviews=[_review_json(False, failing=("relevancy",))],
        revisions=["still weak"],
    )
    agent = _make_agent(_make_retrieval(), llm, max_iterations=1)

    result = await agent.run("What is RAG?", max_iterations=1)

    assert result.iterations == 1
    assert llm.review_calls == 1


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_with_good_answer_passes_review_on_first_try() -> None:
    """A passing review delivers the initial answer without revising."""
    llm = _ScriptedLLM(
        answers=["grounded answer"],
        reviews=[_review_json(True)],
    )
    agent = _make_agent(_make_retrieval(), llm)

    result = await agent.run("What is RAG?")

    assert result.answer == "grounded answer"
    assert result.initial_answer == "grounded answer"
    assert result.revised is False
    assert result.iterations == 1
    assert result.review is not None
    assert result.review.passed is True
    assert llm.revision_calls == 0


# ---------------------------------------------------------------------------
# Graph: accumulation + lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_with_grounding_gap_accumulates_research_results() -> None:
    """A grounding failure loops back and appends a second retrieval pass."""
    llm = _ScriptedLLM(
        answers=["initial", "better"],
        reviews=[
            _review_json(False, failing=("grounding",)),
            _review_json(True),
        ],
    )
    retrieval = _make_retrieval(
        [_make_retrieval_result(), _make_retrieval_result()]
    )
    agent = _make_agent(retrieval, llm)
    graph = build_review_graph(agent, max_iterations=3)

    state = await graph.run("What is RAG?")

    assert state.status == "deliver"
    assert len(state.research_results) == 2  # accumulated, not replaced
    assert retrieval.retrieve.await_count == 2
    assert state.iteration_count == 2


@pytest.mark.asyncio
async def test_graph_run_reaches_delivered_status() -> None:
    """The compiled graph terminates in the deliver state."""
    llm = _ScriptedLLM(reviews=[_review_json(True)])
    graph = build_review_graph(_make_agent(_make_retrieval(), llm))

    assert isinstance(graph, ReviewGraph)
    state = await graph.run("What is RAG?")

    assert state.status == "deliver"
    assert state.iteration_count == 1


@pytest.mark.asyncio
async def test_graph_intake_with_empty_query_raises_agent_graph_error() -> None:
    """Blank queries are rejected at INTAKE."""
    llm = _ScriptedLLM()
    graph = build_review_graph(_make_agent(_make_retrieval(), llm))

    with pytest.raises(AgentGraphError):
        await graph.run("   ")


# ---------------------------------------------------------------------------
# Parsing + state
# ---------------------------------------------------------------------------


def test_parse_review_with_non_json_failure_text_marks_failed() -> None:
    """Keyword fallback catches a non-JSON failure verdict."""
    agent = _make_agent(_make_retrieval(), _ScriptedLLM())

    verdict = agent._parse_review("The answer FAILS to be grounded in context.")

    assert verdict.passed is False


def test_parse_review_with_valid_json_extracts_dimensions() -> None:
    """A well-formed JSON verdict is parsed into dimensions."""
    agent = _make_agent(_make_retrieval(), _ScriptedLLM())

    verdict = agent._parse_review(_review_json(False, failing=("completeness",)))

    assert verdict.passed is False
    assert verdict.failed_dimensions == ["completeness"]
    assert len(verdict.dimensions) == len(REVIEW_DIMENSIONS)


def test_agent_state_accumulation_preserves_all_passes() -> None:
    """AgentState.add_research_results appends rather than replaces."""
    state = AgentState(query="q")
    state.add_research_results([_make_retrieval_result()])
    state.add_research_results([_make_retrieval_result()])
    state.add_research_results([_make_retrieval_result()])

    assert len(state.research_results) == 3
