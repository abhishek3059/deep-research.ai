"""Single-agent self-critique RAG prototype (Phase 2).

Pattern: ``retrieve -> generate -> critique -> revise`` bounded by
:data:`src.config.constants.MAX_ITERATIONS`.  This exists to measure whether a
single agent with a self-critique loop is good enough, before paying for a
multi-agent topology (see ADR-003).

Loop mechanics live in :class:`~src.agents.review_loop.ReviewLoopBase`; this
module supplies the 4-dimension critique vocabulary.

Architecture invariants honored:
- The agent never touches the vector store; it calls
  :class:`~src.retrieval.pipeline.RetrievalPipeline` (invariant #4).
- Generation is delegated to
  :meth:`~src.agents.generation.GenerationPipeline.generate_answer` so the
  prompt/citation contract stays in one place.
- All I/O is ``async`` (invariant #8).
"""

from __future__ import annotations

from src.agents.generation import GenerationPipeline
from src.agents.llm_provider import LLMProvider
from src.agents.review_loop import ReviewDimension, ReviewLoopBase, ReviewVerdict
from src.agents.state import (
    CRITIQUE_DIMENSIONS,
    AgentState,
    CritiqueDimension,
    CritiqueResult,
    SelfCritiqueResult,
)
from src.config.constants import DEFAULT_TOP_K, MAX_ITERATIONS
from src.retrieval.pipeline import RetrievalPipeline

# Markers are used by tests to route a single fake LLM between call types.
CRITIQUE_MARKER = "research critic"
REVISE_MARKER = "research editor"

CRITIQUE_PROMPT = (
    "You are a rigorous research critic. Evaluate the ANSWER using ONLY the "
    "provided CONTEXT and the QUESTION. Assess exactly four dimensions:\n"
    "- faithfulness: every claim is supported by the context (no hallucination)\n"
    "- relevancy: the answer directly addresses the question\n"
    "- accuracy: no factual errors or contradictions\n"
    "- completeness: key information from the context is covered\n\n"
    'Return ONLY a JSON object: {"passed": bool, "dimensions": {'
    '"faithfulness": {"passed": bool, "issue": str}, "relevancy": {...}, '
    '"accuracy": {...}, "completeness": {...}}, "summary": str}. '
    'Set "passed" true only when all four dimensions pass.'
)

REVISE_PROMPT = (
    "You are a research editor. Rewrite the ANSWER so it fixes every issue "
    "raised in the CRITIQUE while staying strictly grounded in the CONTEXT. "
    "Keep inline [Source N] citations. Return only the revised answer text."
)

# Dimensions that, when failed, indicate the context itself is insufficient.
_CONTEXT_GAP_DIMENSIONS = frozenset({"faithfulness", "completeness"})


class SelfCritiqueAgent(ReviewLoopBase):
    """Runs the retrieve -> generate -> critique -> revise loop in-process.

    The public node methods (:meth:`research`, :meth:`critique`, :meth:`revise`)
    operate on :class:`AgentState` so the LangGraph graph in
    :mod:`src.agents.graph` can compose them without duplicating logic.
    """

    REVIEW_STATUS = "critique"
    CONTEXT_GAP_DIMENSIONS = _CONTEXT_GAP_DIMENSIONS
    GUARD_MESSAGE = "Self-critique loop exceeded its guard limit"

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
            generation_pipeline: Answer generator. Defaults to one bound to
                *llm_provider*.
            llm_provider: Unified LLM. Defaults to a new :class:`LLMProvider`.
            max_iterations: Maximum critique iterations (loop guard).
            top_k: Results requested per retrieval pass.
        """
        super().__init__(
            retrieval_pipeline=retrieval_pipeline,
            generation_pipeline=generation_pipeline,
            llm_provider=llm_provider,
            max_iterations=max_iterations,
            top_k=top_k,
        )

    async def critique(self, state: AgentState) -> AgentState:
        """Score the current answer and choose the next status."""
        return await self._review_node(state)

    async def run(
        self,
        query: str,
        max_iterations: int | None = None,
    ) -> SelfCritiqueResult:
        """Run the full loop and return the final answer plus metadata."""
        result = await super().run(query, max_iterations=max_iterations)
        assert isinstance(result, SelfCritiqueResult)  # guaranteed by _build_result
        return result

    def _parse_critique(self, raw: str) -> CritiqueResult:
        """Parse an LLM critique response into a :class:`CritiqueResult`."""
        verdict = self._parse_verdict(raw)
        assert isinstance(verdict, CritiqueResult)  # guaranteed by _make_verdict
        return verdict

    # -- vocabulary hooks -----------------------------------------------------

    def _build_review_messages(self, query: str, answer: str, context: str) -> list[dict[str, str]]:
        """Build the critique prompt messages."""
        user = (
            f"QUESTION:\n{query}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}\n\n"
            "Return only the JSON verdict."
        )
        return [
            {"role": "system", "content": CRITIQUE_PROMPT},
            {"role": "user", "content": user},
        ]

    def _build_revise_messages(
        self, query: str, answer: str, feedback: str, context: str
    ) -> list[dict[str, str]]:
        """Build the revision prompt messages."""
        user = (
            f"QUESTION:\n{query}\n\nCONTEXT:\n{context}\n\n"
            f"ANSWER:\n{answer}\n\nCRITIQUE:\n{feedback}\n\n"
            "Return only the revised answer."
        )
        return [
            {"role": "system", "content": REVISE_PROMPT},
            {"role": "user", "content": user},
        ]

    def _make_dimension(self, name: str, passed: bool, issue: str) -> ReviewDimension:
        """Construct a :class:`CritiqueDimension`."""
        return CritiqueDimension(name=name, passed=passed, issue=issue)

    def _make_verdict(
        self,
        passed: bool,
        dimensions: list[ReviewDimension],
        summary: str,
        raw: str,
    ) -> ReviewVerdict:
        """Construct a :class:`CritiqueResult`."""
        return CritiqueResult(
            passed=passed,
            dimensions=[
                CritiqueDimension(name=d.name, passed=d.passed, issue=d.issue) for d in dimensions
            ],
            summary=summary,
            raw=raw,
        )

    def _store_verdict(self, state: AgentState, verdict: ReviewVerdict) -> None:
        """Record the verdict as ``critique_result`` + human-readable text."""
        assert isinstance(verdict, CritiqueResult)  # guaranteed by _make_verdict
        state.critique_result = verdict
        state.critique = verdict.to_state_text()

    def _read_verdict(self, state: AgentState) -> ReviewVerdict | None:
        """Read the recorded critique, if any."""
        return state.critique_result

    def _feedback_text(self, state: AgentState) -> str:
        """Human-readable critique consumed by the revise prompt."""
        return state.critique

    def _build_result(self, state: AgentState) -> SelfCritiqueResult:
        """Convert terminal state into a :class:`SelfCritiqueResult`."""
        revised = bool(state.revised_answer) and state.revised_answer != state.initial_answer
        return SelfCritiqueResult(
            query=state.query,
            answer=state.current_answer,
            initial_answer=state.initial_answer,
            revised=revised,
            iterations=state.iteration_count,
            critique=state.critique_result,
            sources=state.sources,
            status=state.status,
        )


# Re-exported for callers that want to assert on the dimension set.
__all__ = [
    "CRITIQUE_DIMENSIONS",
    "CRITIQUE_MARKER",
    "CRITIQUE_PROMPT",
    "REVISE_MARKER",
    "REVISE_PROMPT",
    "SelfCritiqueAgent",
]
