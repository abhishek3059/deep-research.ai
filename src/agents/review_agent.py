"""Self-check review agent (Phase 2).

Pattern: ``research -> review -> revise`` bounded by
:data:`src.config.constants.MAX_ITERATIONS`.  A reviewer LLM inspects the
generated answer against the retrieved context on three axes — grounding,
relevancy, completeness — and the agent either delivers the answer or revises
it.  A grounding failure loops back to RESEARCH so more evidence is accumulated
before another attempt.

Architecture invariants honored:
- The agent never touches the vector store; it calls
  :class:`~src.retrieval.pipeline.RetrievalPipeline` (invariant #4).
- Generation is delegated to
  :meth:`~src.agents.generation.GenerationPipeline.generate_answer` so the
  prompt/citation contract stays in one place.
- All I/O is ``async`` (invariant #8).
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

from src.agents.generation import GenerationPipeline
from src.agents.llm_provider import LLMProvider
from src.agents.state import (
    REVIEW_DIMENSIONS,
    AgentState,
    ReviewAgentResult,
    ReviewDimension,
    ReviewResult,
)
from src.config.constants import DEFAULT_TOP_K, MAX_ITERATIONS
from src.retrieval.pipeline import RetrievalMeta, RetrievalPipeline, RetrievalResult
from src.vectorstore.base import SearchResult

logger = structlog.get_logger(__name__)

# Markers are used by tests to route a single fake LLM between call types.
REVIEW_MARKER = "research reviewer"
REVISION_MARKER = "research editor"

REVIEW_PROMPT = (
    "You are a meticulous research reviewer. Evaluate the ANSWER using ONLY the "
    "provided CONTEXT and the QUESTION. Assess exactly three dimensions:\n"
    "- grounding: every claim is supported by the context (no hallucination)\n"
    "- relevancy: the answer directly addresses the question\n"
    "- completeness: key information from the context is covered and no part "
    "of the question is left unanswered\n\n"
    'Return ONLY a JSON object: {"passed": bool, "dimensions": {'
    '"grounding": {"passed": bool, "issue": str}, "relevancy": {...}, '
    '"completeness": {...}}, "summary": str}. '
    'Set "passed" true only when all three dimensions pass.'
)

REVISION_PROMPT = (
    "You are a research editor. Rewrite the ANSWER so it fixes every issue "
    "raised in the REVIEW while staying strictly grounded in the CONTEXT. "
    "Keep inline [Source N] citations and directly answer the question. "
    "Return only the revised answer text."
)

# A grounding failure means the context itself is insufficient — the agent
# should accumulate more evidence rather than merely reword the answer.
_CONTEXT_GAP_DIMENSIONS = frozenset({"grounding"})


class ReviewAgent:
    """Runs the research -> review -> revise loop in-process.

    The public node methods (:meth:`research`, :meth:`review`, :meth:`revise`)
    operate on :class:`AgentState` so the pure-Python graph in
    :mod:`src.agents.graph_skeleton` can compose them without duplicating logic.
    """

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        generation_pipeline: GenerationPipeline | None = None,
        llm_provider: LLMProvider | None = None,
        max_iterations: int = MAX_ITERATIONS,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        """Wire the agent's dependencies.

        Args:
            retrieval_pipeline: Retrieval module entry point (never a raw store).
            generation_pipeline: Answer generator.  Defaults to one bound to
                *llm_provider*.
            llm_provider: Unified LLM.  Defaults to a new :class:`LLMProvider`.
            max_iterations: Maximum review iterations (loop guard).
            top_k: Results requested per retrieval pass.
        """
        self._retrieval = retrieval_pipeline
        self._llm = llm_provider or LLMProvider()
        self._generation = generation_pipeline or GenerationPipeline(llm_provider=self._llm)
        self._max_iterations = max_iterations
        self._top_k = top_k

    # -- graph nodes ----------------------------------------------------------

    async def research(self, state: AgentState) -> AgentState:
        """Retrieve context (accumulating) and (re)generate the answer."""
        result = await self._retrieval.retrieve(state.query, top_k=self._top_k)
        state.add_research_results([result])
        state.decomposed_queries = self._collect_queries(state.research_results)

        merged = self._merge_results(state.research_results)
        generation = await self._generation.generate_answer(state.query, merged)
        state.initial_answer = str(generation.get("answer", ""))
        state.sources = self._extract_sources(merged)
        state.status = "review"

        logger.info(
            "Research pass complete",
            query=state.query[:80],
            passes=len(state.research_results),
            context_chunks=len(merged.results),
        )
        return state

    async def review(self, state: AgentState) -> AgentState:
        """Self-check the current answer and choose the next status."""
        context = self._format_context(self._merge_results(state.research_results))
        messages = self._review_messages(state.query, state.current_answer, context)
        raw = await self._llm.generate(messages, temperature=0.0)

        state.review_result = self._parse_review(raw)
        state.review_notes = state.review_result.to_notes()
        state.iteration_count += 1
        state.status = self._route_after_review(state)

        logger.info(
            "Review complete",
            iteration=state.iteration_count,
            passed=state.review_result.passed,
            next_status=state.status,
        )
        return state

    async def revise(self, state: AgentState) -> AgentState:
        """Rewrite the answer to address the recorded review notes."""
        context = self._format_context(self._merge_results(state.research_results))
        messages = self._revision_messages(
            state.query, state.current_answer, state.review_notes, context
        )
        state.revised_answer = await self._llm.generate(messages, temperature=0.2)
        state.status = "review"
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
        if state.status == "review":
            return await self.review(state)
        if state.status == "revise":
            return await self.revise(state)
        if state.status == "deliver":
            return state
        raise ValueError(f"Unknown agent status: {state.status!r}")

    async def run(
        self,
        query: str,
        max_iterations: int | None = None,
    ) -> ReviewAgentResult:
        """Run the full loop and return the final answer plus review metadata.

        Args:
            query: User question.
            max_iterations: Overrides the instance loop guard when provided.

        Returns:
            A :class:`ReviewAgentResult` describing the delivered answer.
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
                raise RuntimeError("Review loop exceeded its guard limit")

        return self._to_result(state)

    # -- routing --------------------------------------------------------------

    def _route_after_review(self, state: AgentState) -> str:
        """Decide the next status, enforcing the loop guard."""
        verdict = state.review_result
        if verdict is None or verdict.passed:
            return "deliver"
        if state.iteration_count >= state.max_iterations:
            return "deliver"  # loop guard: deliver the best answer we have
        if _CONTEXT_GAP_DIMENSIONS & set(verdict.failed_dimensions):
            return "research"  # missing grounding -> accumulate more evidence
        return "revise"

    # -- result assembly ------------------------------------------------------

    def _to_result(self, state: AgentState) -> ReviewAgentResult:
        """Convert terminal state into the boundary result object."""
        revised = bool(state.revised_answer) and state.revised_answer != state.initial_answer
        return ReviewAgentResult(
            query=state.query,
            answer=state.current_answer,
            initial_answer=state.initial_answer,
            revised=revised,
            iterations=state.iteration_count,
            review=state.review_result,
            sources=state.sources,
            status=state.status,
        )

    # -- LLM message builders -------------------------------------------------

    def _review_messages(
        self, query: str, answer: str, context: str
    ) -> list[dict[str, str]]:
        """Build the self-check review prompt messages."""
        user = (
            f"QUESTION:\n{query}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}\n\n"
            "Return only the JSON verdict."
        )
        return [
            {"role": "system", "content": REVIEW_PROMPT},
            {"role": "user", "content": user},
        ]

    def _revision_messages(
        self, query: str, answer: str, notes: str, context: str
    ) -> list[dict[str, str]]:
        """Build the revision prompt messages."""
        user = (
            f"QUESTION:\n{query}\n\nCONTEXT:\n{context}\n\n"
            f"ANSWER:\n{answer}\n\nREVIEW:\n{notes}\n\n"
            "Return only the revised answer."
        )
        return [
            {"role": "system", "content": REVISION_PROMPT},
            {"role": "user", "content": user},
        ]

    # -- parsing --------------------------------------------------------------

    def _parse_review(self, raw: str) -> ReviewResult:
        """Parse an LLM review response into a :class:`ReviewResult`.

        Falls back to keyword heuristics when the model ignores the JSON
        instruction rather than raising.
        """
        payload = self._extract_json(raw)
        if payload is None:
            return self._fallback_review(raw)

        dimensions = self._normalize_dimensions(payload.get("dimensions"))
        if "passed" in payload:
            passed = bool(payload["passed"])
        else:
            passed = all(d.passed for d in dimensions) if dimensions else True
        return ReviewResult(
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

    @staticmethod
    def _normalize_dimensions(value: Any) -> list[ReviewDimension]:
        """Coerce dict- or list-shaped dimension payloads to models."""
        dimensions: list[ReviewDimension] = []
        if isinstance(value, dict):
            entries = [(str(name), spec) for name, spec in value.items()]
        elif isinstance(value, list):
            entries = [
                (str(entry.get("name", "")), entry)
                for entry in value
                if isinstance(entry, dict)
            ]
        else:
            return dimensions

        for name, spec in entries:
            if isinstance(spec, dict):
                dimensions.append(
                    ReviewDimension(
                        name=name,
                        passed=bool(spec.get("passed", False)),
                        issue=str(spec.get("issue", "")),
                    )
                )
            else:
                dimensions.append(ReviewDimension(name=name, passed=bool(spec), issue=""))
        return dimensions

    @staticmethod
    def _fallback_review(raw: str) -> ReviewResult:
        """Keyword fallback when the response is not parseable JSON."""
        lowered = raw.lower()
        has_fail = re.search(r"\bfail(?:ed|s|ure|ing)?\b", lowered) is not None
        has_pass = re.search(r"\bpass(?:ed|es|ing)?\b", lowered) is not None
        passed = not has_fail or has_pass
        return ReviewResult(passed=passed, summary=raw.strip()[:500], raw=raw)

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


# Re-exported for callers that want to assert on the dimension set.
__all__ = [
    "REVIEW_DIMENSIONS",
    "REVIEW_MARKER",
    "REVIEW_PROMPT",
    "REVISION_MARKER",
    "REVISION_PROMPT",
    "ReviewAgent",
]
