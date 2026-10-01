"""Self-check review agent (Phase 2).

Pattern: ``research -> review -> revise`` bounded by
:data:`src.config.constants.MAX_ITERATIONS`.  A reviewer LLM inspects the
generated answer against the retrieved context on three axes — grounding,
relevancy, completeness — and the agent either delivers the answer or revises
it.  A grounding failure loops back to RESEARCH so more evidence is accumulated
before another attempt.

Loop mechanics live in :class:`~src.agents.review_loop.ReviewLoopBase`; this
module supplies the 3-dimension review vocabulary.

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
from src.agents.review_loop import ReviewDimension as ReviewDimensionProto
from src.agents.review_loop import ReviewLoopBase, ReviewVerdict
from src.agents.state import (
    REVIEW_DIMENSIONS,
    AgentState,
    ReviewAgentResult,
    ReviewDimension,
    ReviewResult,
)
from src.config.constants import DEFAULT_TOP_K, MAX_ITERATIONS
from src.retrieval.pipeline import RetrievalPipeline

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


class ReviewAgent(ReviewLoopBase):
    """Runs the research -> review -> revise loop in-process.

    The public node methods (:meth:`research`, :meth:`review`, :meth:`revise`)
    operate on :class:`AgentState` so the pure-Python graph in
    :mod:`src.agents.graph_skeleton` can compose them without duplicating logic.
    """

    REVIEW_STATUS = "review"
    CONTEXT_GAP_DIMENSIONS = _CONTEXT_GAP_DIMENSIONS
    GUARD_MESSAGE = "Review loop exceeded its guard limit"

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
        super().__init__(
            retrieval_pipeline=retrieval_pipeline,
            generation_pipeline=generation_pipeline,
            llm_provider=llm_provider,
            max_iterations=max_iterations,
            top_k=top_k,
        )

    async def review(self, state: AgentState) -> AgentState:
        """Self-check the current answer and choose the next status."""
        return await self._review_node(state)

    async def run(
        self,
        query: str,
        max_iterations: int | None = None,
    ) -> ReviewAgentResult:
        """Run the full loop and return the final answer plus review metadata."""
        result = await super().run(query, max_iterations=max_iterations)
        assert isinstance(result, ReviewAgentResult)  # guaranteed by _build_result
        return result

    def _parse_review(self, raw: str) -> ReviewResult:
        """Parse an LLM review response into a :class:`ReviewResult`."""
        verdict = self._parse_verdict(raw)
        assert isinstance(verdict, ReviewResult)  # guaranteed by _make_verdict
        return verdict

    # -- vocabulary hooks -----------------------------------------------------

    def _build_review_messages(self, query: str, answer: str, context: str) -> list[dict[str, str]]:
        """Build the self-check review prompt messages."""
        user = (
            f"QUESTION:\n{query}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}\n\n"
            "Return only the JSON verdict."
        )
        return [
            {"role": "system", "content": REVIEW_PROMPT},
            {"role": "user", "content": user},
        ]

    def _build_revise_messages(
        self, query: str, answer: str, feedback: str, context: str
    ) -> list[dict[str, str]]:
        """Build the revision prompt messages."""
        user = (
            f"QUESTION:\n{query}\n\nCONTEXT:\n{context}\n\n"
            f"ANSWER:\n{answer}\n\nREVIEW:\n{feedback}\n\n"
            "Return only the revised answer."
        )
        return [
            {"role": "system", "content": REVISION_PROMPT},
            {"role": "user", "content": user},
        ]

    def _make_dimension(self, name: str, passed: bool, issue: str) -> ReviewDimension:
        """Construct a :class:`ReviewDimension`."""
        return ReviewDimension(name=name, passed=passed, issue=issue)

    def _make_verdict(
        self,
        passed: bool,
        dimensions: list[ReviewDimensionProto],
        summary: str,
        raw: str,
    ) -> ReviewVerdict:
        """Construct a :class:`ReviewResult`."""
        return ReviewResult(
            passed=passed,
            dimensions=[
                ReviewDimension(name=d.name, passed=d.passed, issue=d.issue) for d in dimensions
            ],
            summary=summary,
            raw=raw,
        )

    def _store_verdict(self, state: AgentState, verdict: ReviewVerdict) -> None:
        """Record the verdict as ``review_result`` + human-readable notes."""
        assert isinstance(verdict, ReviewResult)  # guaranteed by _make_verdict
        state.review_result = verdict
        state.review_notes = verdict.to_notes()

    def _read_verdict(self, state: AgentState) -> ReviewVerdict | None:
        """Read the recorded review, if any."""
        return state.review_result

    def _feedback_text(self, state: AgentState) -> str:
        """Human-readable notes consumed by the revise prompt."""
        return state.review_notes

    def _build_result(self, state: AgentState) -> ReviewAgentResult:
        """Convert terminal state into a :class:`ReviewAgentResult`."""
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


# Re-exported for callers that want to assert on the dimension set.
__all__ = [
    "REVIEW_DIMENSIONS",
    "REVIEW_MARKER",
    "REVIEW_PROMPT",
    "REVISION_MARKER",
    "REVISION_PROMPT",
    "ReviewAgent",
]
