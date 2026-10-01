"""Shared retrieve -> review -> revise loop mechanics (Phase 2).

:class:`ReviewLoopBase` owns everything the self-critique loop does *except*
the review vocabulary: prompts, dimension sets, verdict models, and the state
fields they are stored in. Subclasses
(:class:`~src.agents.self_critique.SelfCritiqueAgent` with its 4-dimension
critique, :class:`~src.agents.review_agent.ReviewAgent` with its 3-dimension
review) supply those pieces through small factory hooks.

Extracted to eliminate the ~80% duplication the 2026-09-16 architect review
flagged between ``self_critique.py`` and ``review_agent.py``. Public APIs of
both subclasses are unchanged.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Protocol

import structlog
from pydantic import BaseModel

from src.agents.generation import GenerationPipeline
from src.agents.llm_provider import LLMProvider
from src.agents.prompts import NO_COVERAGE_RESPONSE
from src.agents.state import AgentState
from src.config.constants import DEFAULT_TOP_K, MAX_ITERATIONS
from src.retrieval.pipeline import RetrievalMeta, RetrievalPipeline, RetrievalResult
from src.vectorstore.base import SearchResult

logger = structlog.get_logger(__name__)


class ReviewDimension(Protocol):
    """Structural type for a single scored axis (critique or review)."""

    name: str
    passed: bool
    issue: str


class ReviewVerdict(Protocol):
    """Structural type for a parsed review verdict."""

    passed: bool

    @property
    def failed_dimensions(self) -> list[str]: ...


class ReviewLoopBase(ABC):
    """Retrieve -> review -> revise loop with accumulation semantics.

    Subclasses define the review vocabulary via :attr:`REVIEW_STATUS`,
    :attr:`CONTEXT_GAP_DIMENSIONS`, and the factory/state hooks below.
    """

    #: Status written after research/revise (``"critique"`` or ``"review"``).
    REVIEW_STATUS: str = "review"
    #: Failed dimensions that mean "more context needed" (-> research).
    CONTEXT_GAP_DIMENSIONS: frozenset[str] = frozenset()
    #: Error raised when the step guard trips.
    GUARD_MESSAGE: str = "Review loop exceeded its guard limit"

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        generation_pipeline: GenerationPipeline | None = None,
        llm_provider: LLMProvider | None = None,
        max_iterations: int = MAX_ITERATIONS,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        """Wire the loop's dependencies.

        Args:
            retrieval_pipeline: Retrieval module entry point (never a raw store).
            generation_pipeline: Answer generator. Defaults to one bound to
                *llm_provider*.
            llm_provider: Unified LLM. Defaults to a new :class:`LLMProvider`.
            max_iterations: Maximum review iterations (loop guard).
            top_k: Results requested per retrieval pass.
        """
        self._retrieval = retrieval_pipeline
        self._llm = llm_provider or LLMProvider()
        self._generation = generation_pipeline or GenerationPipeline(llm_provider=self._llm)
        self._max_iterations = max_iterations
        self._top_k = top_k

    # -- vocabulary hooks (subclass-provided) ---------------------------------

    @abstractmethod
    def _build_review_messages(self, query: str, answer: str, context: str) -> list[dict[str, str]]:
        """Build the verdict prompt messages."""
        ...

    @abstractmethod
    def _build_revise_messages(
        self, query: str, answer: str, feedback: str, context: str
    ) -> list[dict[str, str]]:
        """Build the revision prompt messages."""
        ...

    @abstractmethod
    def _make_dimension(self, name: str, passed: bool, issue: str) -> ReviewDimension:
        """Construct the subclass's dimension model."""
        ...

    @abstractmethod
    def _make_verdict(
        self,
        passed: bool,
        dimensions: list[ReviewDimension],
        summary: str,
        raw: str,
    ) -> ReviewVerdict:
        """Construct the subclass's verdict model."""
        ...

    @abstractmethod
    def _store_verdict(self, state: AgentState, verdict: ReviewVerdict) -> None:
        """Record the verdict in the subclass's state fields."""
        ...

    @abstractmethod
    def _read_verdict(self, state: AgentState) -> ReviewVerdict | None:
        """Read the recorded verdict, if any."""
        ...

    @abstractmethod
    def _feedback_text(self, state: AgentState) -> str:
        """Human-readable verdict text consumed by the revise prompt."""
        ...

    @abstractmethod
    def _build_result(self, state: AgentState) -> BaseModel:
        """Convert terminal state into the subclass's boundary result object."""
        ...

    # -- graph nodes ----------------------------------------------------------

    async def research(self, state: AgentState) -> AgentState:
        """Retrieve context (accumulating) and (re)generate the answer.

        When the retrieval coverage floor trips (no results), short-circuit
        to an honest templated answer: no LLM call is made, no sources are
        claimed, and the state delivers directly.
        """
        result = await self._retrieval.retrieve(state.query, top_k=self._top_k)
        if not result.results:
            state.no_coverage = True
            state.initial_answer = NO_COVERAGE_RESPONSE
            state.sources = []
            state.status = "deliver"
            logger.info("No coverage — delivering honest short-circuit", query=state.query[:80])
            return state
        state.add_research_results([result])
        state.decomposed_queries = self._collect_queries(state.research_results)

        merged = self._merge_results(state.research_results)
        generation = await self._generation.generate_answer(state.query, merged)
        state.initial_answer = str(generation.get("answer", ""))
        state.sources = self._extract_sources(merged)
        state.status = self.REVIEW_STATUS

        logger.info(
            "Research pass complete",
            query=state.query[:80],
            passes=len(state.research_results),
            context_chunks=len(merged.results),
        )
        return state

    async def _review_node(self, state: AgentState) -> AgentState:
        """Score the current answer and choose the next status."""
        if state.no_coverage or state.status == "deliver":
            # Unconditional graph edges can land here after a short-circuit;
            # there is nothing to critique.
            return state
        context = self._format_context(self._merge_results(state.research_results))
        messages = self._build_review_messages(state.query, state.current_answer, context)
        raw = await self._llm.generate(messages, temperature=0.0)

        verdict = self._parse_verdict(raw)
        self._store_verdict(state, verdict)
        state.iteration_count += 1
        state.status = self._route_after(state)

        logger.info(
            "Review complete",
            iteration=state.iteration_count,
            passed=verdict.passed,
            next_status=state.status,
        )
        return state

    async def revise(self, state: AgentState) -> AgentState:
        """Rewrite the answer to address the recorded verdict."""
        if state.no_coverage or state.status == "deliver":
            return state
        context = self._format_context(self._merge_results(state.research_results))
        messages = self._build_revise_messages(
            state.query, state.current_answer, self._feedback_text(state), context
        )
        state.revised_answer = await self._llm.generate(messages, temperature=0.2)
        state.status = self.REVIEW_STATUS
        return state

    # -- orchestration --------------------------------------------------------

    async def step(self, state: AgentState) -> AgentState:
        """Execute one node based on ``state.status``.

        Args:
            state: Current graph state.

        Returns:
            The state after the node ran.

        Raises:
            ValueError: If the status is not a known node.
        """
        if state.status == "intake":
            state.status = "research"
            return state
        if state.status == "research":
            return await self.research(state)
        if state.status == self.REVIEW_STATUS:
            return await self._review_node(state)
        if state.status == "revise":
            return await self.revise(state)
        if state.status == "deliver":
            return state
        raise ValueError(f"Unknown agent status: {state.status!r}")

    async def run(
        self,
        query: str,
        max_iterations: int | None = None,
    ) -> BaseModel:
        """Run the full loop and return the final answer plus metadata.

        Args:
            query: User question.
            max_iterations: Overrides the instance loop guard when provided.

        Returns:
            The subclass's boundary result object describing the delivered answer.
        """
        state = AgentState(
            query=query,
            max_iterations=max_iterations or self._max_iterations,
        )
        guard = state.max_iterations * 4 + 10
        steps = 0

        while state.status != "deliver":
            if state.iteration_count >= state.max_iterations:
                state.status = "deliver"
                break
            state = await self.step(state)
            steps += 1
            if steps > guard:
                raise RuntimeError(self.GUARD_MESSAGE)

        return self._build_result(state)

    # -- routing --------------------------------------------------------------

    def _route_after(self, state: AgentState) -> str:
        """Decide the next status, enforcing the loop guard."""
        verdict = self._read_verdict(state)
        if verdict is None or verdict.passed:
            return "deliver"
        if state.iteration_count >= state.max_iterations:
            return "deliver"  # loop guard: deliver the best answer we have
        if set(verdict.failed_dimensions) & set(self.CONTEXT_GAP_DIMENSIONS):
            return "research"  # missing grounding -> accumulate more context
        return "revise"

    # -- parsing --------------------------------------------------------------

    def _parse_verdict(self, raw: str) -> ReviewVerdict:
        """Parse an LLM verdict response into the subclass's verdict model.

        Falls back to keyword heuristics when the model ignores the JSON
        instruction rather than raising.
        """
        payload = self._extract_json(raw)
        if payload is None:
            return self._fallback_verdict(raw)

        dimensions = self._normalize_dimensions(payload.get("dimensions"))
        if "passed" in payload:
            passed = bool(payload["passed"])
        else:
            passed = all(d.passed for d in dimensions) if dimensions else True
        return self._make_verdict(
            passed=passed,
            dimensions=dimensions,
            summary=str(payload.get("summary", "")),
            raw=raw,
        )

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any] | None:
        """Return the first JSON object embedded in *raw*, if any."""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match is None:
            return None
        try:
            data = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _normalize_dimensions(self, value: Any) -> list[ReviewDimension]:
        """Coerce dict- or list-shaped dimension payloads to models."""
        dimensions: list[ReviewDimension] = []
        if isinstance(value, dict):
            entries = [(str(name), spec) for name, spec in value.items()]
        elif isinstance(value, list):
            entries = [
                (str(entry.get("name", "")), entry) for entry in value if isinstance(entry, dict)
            ]
        else:
            return dimensions

        for name, spec in entries:
            if isinstance(spec, dict):
                dimensions.append(
                    self._make_dimension(
                        name=name,
                        passed=bool(spec.get("passed", False)),
                        issue=str(spec.get("issue", "")),
                    )
                )
            else:
                dimensions.append(self._make_dimension(name=name, passed=bool(spec), issue=""))
        return dimensions

    def _fallback_verdict(self, raw: str) -> ReviewVerdict:
        """Keyword fallback when the response is not parseable JSON."""
        lowered = raw.lower()
        has_fail = re.search(r"\bfail(?:ed|s|ure|ing)?\b", lowered) is not None
        has_pass = re.search(r"\bpass(?:ed|es|ing)?\b", lowered) is not None
        passed = not has_fail or has_pass
        return self._make_verdict(passed=passed, dimensions=[], summary=raw.strip()[:500], raw=raw)

    # -- retrieval merging ----------------------------------------------------

    def _merge_results(self, results: list[RetrievalResult]) -> RetrievalResult:
        """Flatten accumulated passes into one de-duplicated RetrievalResult."""
        seen: set[str] = set()
        merged: list[SearchResult] = []
        dense = sparse = 0
        latency = 0.0
        reranked = False
        strategy = "dense"

        for result in results:
            meta = result.retrieval_metadata
            dense += meta.dense_results
            sparse += meta.sparse_results
            latency += meta.latency_ms
            reranked = reranked or meta.reranked
            if meta.strategy == "hybrid":
                strategy = "hybrid"
            for search_result in result.results:
                if search_result.id not in seen:
                    seen.add(search_result.id)
                    merged.append(search_result)

        query = results[0].query if results else ""
        meta = RetrievalMeta(
            strategy=strategy,
            dense_results=dense,
            sparse_results=sparse,
            reranked=reranked,
            latency_ms=round(latency, 2),
        )
        return RetrievalResult(
            query=query,
            expanded_queries=self._collect_queries(results),
            results=merged,
            retrieval_metadata=meta,
        )

    @staticmethod
    def _collect_queries(results: list[RetrievalResult]) -> list[str]:
        """Collect unique expanded queries across retrieval passes."""
        queries: list[str] = []
        for result in results:
            for query in result.expanded_queries:
                if query not in queries:
                    queries.append(query)
        return queries

    @staticmethod
    def _format_context(result: RetrievalResult) -> str:
        """Render retrieved chunks as a numbered context block."""
        parts = [f"[Source {i}] {sr.text}" for i, sr in enumerate(result.results, 1)]
        return "\n\n".join(parts) if parts else "No relevant context found."

    @staticmethod
    def _extract_sources(result: RetrievalResult) -> list[dict[str, str]]:
        """Pull lightweight source references from merged results."""
        return [
            {
                "id": sr.id,
                "source": sr.metadata.source,
                "source_type": sr.metadata.source_type,
                "score": str(sr.score),
                "reference": f"[Source {i}]",
            }
            for i, sr in enumerate(result.results, 1)
        ]
