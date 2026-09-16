"""Unit tests for the self-critique agent and state machine.

All LLM and retrieval I/O is mocked; these tests assert the control-flow
contracts from ADR-003 (loop guard, accumulation, critique routing).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from src.agents.exceptions import AgentGraphError
from src.agents.generation import GenerationPipeline
from src.agents.graph import SelfCritiqueGraph, build_graph
from src.agents.memory import ConversationMemory
from src.agents.self_critique import (
    CRITIQUE_MARKER,
    REVISE_MARKER,
    SelfCritiqueAgent,
)
from src.agents.state import CRITIQUE_DIMENSIONS, AgentState
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


def _critique_json(passed: bool, failing: tuple[str, ...] = ()) -> str:
    """Build a well-formed critique verdict JSON string."""
    dimensions = {
        name: {"passed": name not in failing, "issue": "" if name not in failing else f"bad {name}"}
        for name in CRITIQUE_DIMENSIONS
    }
    return json.dumps(
        {"passed": passed, "dimensions": dimensions, "summary": "verdict"}
    )


class _ScriptedLLM:
    """Fake LLM that routes calls by inspecting the system prompt marker."""

    def __init__(
        self,
        answers: list[str] | None = None,
        critiques: list[str] | None = None,
        revisions: list[str] | None = None,
    ) -> None:
        self._answers = answers or ["initial answer"]
        self._critiques = critiques or [_critique_json(True)]
        self._revisions = revisions or ["revised answer"]
        self.generation_calls = 0
        self.critique_calls = 0
        self.revise_calls = 0

    @staticmethod
    def _pick(sequence: list[str], index: int) -> str:
        return sequence[min(index, len(sequence) - 1)]

    async def generate(
        self, messages: list[dict[str, str]], temperature: float = 0.3
    ) -> str:
        system = messages[0]["content"]
        if CRITIQUE_MARKER in system:
            value = self._pick(self._critiques, self.critique_calls)
            self.critique_calls += 1
            return value
        if REVISE_MARKER in system:
            value = self._pick(self._revisions, self.revise_calls)
            self.revise_calls += 1
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


def _make_agent(
    retrieval: AsyncMock, llm: _ScriptedLLM, **kwargs: object
) -> SelfCritiqueAgent:
    generation = GenerationPipeline(
        llm_provider=llm, memory=ConversationMemory()  # type: ignore[arg-type]
    )
    return SelfCritiqueAgent(
        retrieval,
        generation_pipeline=generation,
        llm_provider=llm,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_with_good_answer_skips_revision() -> None:
    """A passing critique delivers the initial answer without revising."""
    llm = _ScriptedLLM(answers=["grounded answer"], critiques=[_critique_json(True)])
    agent = _make_agent(_make_retrieval(), llm)

    result = await agent.run("What is RAG?")

    assert result.answer == "grounded answer"
    assert result.initial_answer == "grounded answer"
    assert result.revised is False
    assert result.iterations == 1
    assert result.critique is not None
    assert result.critique.passed is True
    assert llm.revise_calls == 0


# ---------------------------------------------------------------------------
# Critique identification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_critique_with_wrong_answer_flags_grounding_issues() -> None:
    """A hallucinated answer is flagged and routes back to research."""
    llm = _ScriptedLLM(
        critiques=[_critique_json(False, failing=("faithfulness", "accuracy"))]
    )
    agent = _make_agent(_make_retrieval(), llm)
    state = AgentState(query="What is RAG?")
    state.initial_answer = "RAG was invented on Mars in 1987."
    state.add_research_results([_make_retrieval_result()])

    state = await agent.critique(state)

    assert state.critique_result is not None
    assert state.critique_result.passed is False
    assert "faithfulness" in state.critique_result.failed_dimensions
    assert "accuracy" in state.critique_result.failed_dimensions
    assert "FAIL" in state.critique
    assert state.iteration_count == 1
    # Faithfulness is a context gap -> accumulate more evidence.
    assert state.status == "research"


@pytest.mark.asyncio
async def test_critique_with_shallow_answer_routes_to_revise() -> None:
    """Relevancy/accuracy failures route to REVISE, not re-retrieval."""
    llm = _ScriptedLLM(
        critiques=[_critique_json(False, failing=("relevancy", "accuracy"))]
    )
    agent = _make_agent(_make_retrieval(), llm)
    state = AgentState(query="What is RAG?")
    state.initial_answer = "something off topic"
    state.add_research_results([_make_retrieval_result()])

    state = await agent.critique(state)

    assert state.status == "revise"


# ---------------------------------------------------------------------------
# Revision
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_with_weak_answer_produces_revised_answer() -> None:
    """A failed first critique triggers a revision that is delivered."""
    llm = _ScriptedLLM(
        answers=["weak answer"],
        critiques=[
            _critique_json(False, failing=("relevancy", "accuracy")),
            _critique_json(True),
        ],
        revisions=["improved answer"],
    )
    agent = _make_agent(_make_retrieval(), llm)

    result = await agent.run("What is RAG?")

    assert result.initial_answer == "weak answer"
    assert result.revised is True
    assert result.answer == "improved answer"
    assert result.iterations == 2
    assert llm.revise_calls == 1
    assert result.critique is not None
    assert result.critique.passed is True


# ---------------------------------------------------------------------------
# Loop guard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_with_persistent_failures_respects_loop_guard() -> None:
    """An answer that never passes stops at max_iterations, not forever."""
    llm = _ScriptedLLM(
        answers=["always weak"],
        critiques=[_critique_json(False, failing=("relevancy", "accuracy"))],
        revisions=["still weak"],
    )
    agent = _make_agent(_make_retrieval(), llm, max_iterations=3)

    result = await agent.run("What is RAG?")

    assert result.iterations == 3
    assert llm.critique_calls == 3
    assert llm.revise_calls <= 3
    assert result.critique is not None
    assert result.critique.passed is False


@pytest.mark.asyncio
async def test_run_with_custom_max_iterations_bounds_critiques() -> None:
    """The guard honors an explicit max_iterations override."""
    llm = _ScriptedLLM(
        critiques=[_critique_json(False, failing=("relevancy",))],
        revisions=["still weak"],
    )
    agent = _make_agent(_make_retrieval(), llm, max_iterations=1)

    result = await agent.run("What is RAG?", max_iterations=1)

    assert result.iterations == 1
    assert llm.critique_calls == 1


# ---------------------------------------------------------------------------
# Graph: accumulation + lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_graph_with_context_gap_accumulates_research_results() -> None:
    """A faithfulness failure loops back and appends a second retrieval pass."""
    llm = _ScriptedLLM(
        answers=["initial", "better"],
        critiques=[
            _critique_json(False, failing=("faithfulness",)),
            _critique_json(True),
        ],
    )
    retrieval = _make_retrieval(
        [_make_retrieval_result(), _make_retrieval_result()]
    )
    agent = _make_agent(retrieval, llm)
    graph = build_graph(agent, max_iterations=3)

    state = await graph.run("What is RAG?")

    assert state.status == "deliver"
    assert len(state.research_results) == 2  # accumulated, not replaced
    assert retrieval.retrieve.await_count == 2
    assert state.iteration_count == 2


@pytest.mark.asyncio
async def test_graph_run_reaches_delivered_status() -> None:
    """The compiled graph terminates in the deliver state."""
    llm = _ScriptedLLM(critiques=[_critique_json(True)])
    graph = build_graph(_make_agent(_make_retrieval(), llm))

    assert isinstance(graph, SelfCritiqueGraph)
    state = await graph.run("What is RAG?")

    assert state.status == "deliver"
    assert state.iteration_count == 1


@pytest.mark.asyncio
async def test_graph_intake_with_empty_query_raises_agent_graph_error() -> None:
    """Blank queries are rejected at INTAKE."""
    llm = _ScriptedLLM()
    graph = build_graph(_make_agent(_make_retrieval(), llm))

    with pytest.raises(AgentGraphError):
        await graph.run("   ")


# ---------------------------------------------------------------------------
# Parsing + merging
# ---------------------------------------------------------------------------

def test_parse_critique_with_non_json_failure_text_marks_failed() -> None:
    """Keyword fallback catches a non-JSON failure verdict."""
    agent = _make_agent(_make_retrieval(), _ScriptedLLM())

    verdict = agent._parse_critique("The answer FAILS to be grounded in context.")

    assert verdict.passed is False


def test_parse_critique_with_valid_json_extracts_dimensions() -> None:
    """A well-formed JSON verdict is parsed into dimensions."""
    agent = _make_agent(_make_retrieval(), _ScriptedLLM())

    verdict = agent._parse_critique(_critique_json(False, failing=("completeness",)))

    assert verdict.passed is False
    assert verdict.failed_dimensions == ["completeness"]
    assert len(verdict.dimensions) == len(CRITIQUE_DIMENSIONS)


def test_merge_results_with_overlapping_chunks_deduplicates() -> None:
    """Merging accumulated passes de-duplicates chunks by id."""
    agent = _make_agent(_make_retrieval(), _ScriptedLLM())
    first = _make_retrieval_result(n=2)  # chunk_1, chunk_2
    second = _make_retrieval_result(n=3)  # chunk_1, chunk_2, chunk_3

    merged = agent._merge_results([first, second])

    assert [sr.id for sr in merged.results] == ["chunk_1", "chunk_2", "chunk_3"]


def test_agent_state_accumulation_preserves_all_passes() -> None:
    """AgentState.add_research_results appends rather than replaces."""
    state = AgentState(query="q")
    state.add_research_results([_make_retrieval_result()])
    state.add_research_results([_make_retrieval_result()])
    state.add_research_results([_make_retrieval_result()])

    assert len(state.research_results) == 3
