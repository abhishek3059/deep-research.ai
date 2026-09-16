"""Pydantic state models for the Phase 2 self-critique agent graph.

Per ADR-003 the graph state is a Pydantic model (not a ``TypedDict``) with
**accumulation semantics** for loop-backs: :attr:`AgentState.research_results`
grows across iterations instead of being replaced, so a re-research pass
augments the context rather than discarding earlier evidence.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.config.constants import MAX_ITERATIONS
from src.retrieval.pipeline import RetrievalResult

# Valid values for AgentState.status, in transition order.
# intake -> research -> critique -> revise -> deliver
AGENT_STATUSES: tuple[str, ...] = ("intake", "research", "critique", "revise", "deliver")

# Dimensions the self-critique checks.  Kept in one place so the prompt and
# the parser can never drift apart.
CRITIQUE_DIMENSIONS: tuple[str, ...] = (
    "faithfulness",
    "relevancy",
    "accuracy",
    "completeness",
)

# Dimensions the Phase 2 self-check *review* measures.  Kept in one place so the
# review prompt and the response parser can never drift apart.
REVIEW_DIMENSIONS: tuple[str, ...] = (
    "grounding",
    "relevancy",
    "completeness",
)


class CritiqueDimension(BaseModel):
    """A single scored axis of the self-critique."""

    name: str
    passed: bool
    issue: str = ""


class CritiqueResult(BaseModel):
    """Structured verdict produced by the critique node."""

    passed: bool
    dimensions: list[CritiqueDimension] = Field(default_factory=list)
    summary: str = ""
    raw: str = ""

    @property
    def failed_dimensions(self) -> list[str]:
        """Names of the dimensions that did not pass."""
        return [d.name for d in self.dimensions if not d.passed]

    def to_state_text(self) -> str:
        """Render the verdict as the human-readable ``AgentState.critique``."""
        lines = [f"Verdict: {'PASS' if self.passed else 'FAIL'}"]
        for dim in self.dimensions:
            marker = "ok" if dim.passed else "FAIL"
            suffix = f" - {dim.issue}" if dim.issue else ""
            lines.append(f"- {dim.name}: {marker}{suffix}")
        if self.summary:
            lines.append(f"Summary: {self.summary}")
        return "\n".join(lines)


class ReviewDimension(BaseModel):
    """A single scored axis of the answer self-check."""

    name: str
    passed: bool
    issue: str = ""


class ReviewResult(BaseModel):
    """Structured verdict produced by the self-check REVIEW node."""

    passed: bool
    dimensions: list[ReviewDimension] = Field(default_factory=list)
    summary: str = ""
    raw: str = ""

    @property
    def failed_dimensions(self) -> list[str]:
        """Names of the dimensions that did not pass."""
        return [d.name for d in self.dimensions if not d.passed]

    def to_notes(self) -> str:
        """Render the verdict as the human-readable ``AgentState.review_notes``."""
        lines = [f"Verdict: {'PASS' if self.passed else 'FAIL'}"]
        for dim in self.dimensions:
            marker = "ok" if dim.passed else "FAIL"
            suffix = f" - {dim.issue}" if dim.issue else ""
            lines.append(f"- {dim.name}: {marker}{suffix}")
        if self.summary:
            lines.append(f"Summary: {self.summary}")
        return "\n".join(lines)


class AgentState(BaseModel):
    """State threaded through INTAKE -> RESEARCH -> CRITIQUE/REVIEW -> REVISE -> DELIVER.

    ``research_results`` accumulates: each RESEARCH pass appends a new
    :class:`RetrievalResult` so loop-backs augment rather than replace context.
    """

    query: str
    decomposed_queries: list[str] = Field(default_factory=list)
    research_results: list[RetrievalResult] = Field(default_factory=list)
    initial_answer: str = ""
    critique: str = ""
    revised_answer: str = ""
    iteration_count: int = 0
    max_iterations: int = MAX_ITERATIONS
    status: str = "intake"  # one of AGENT_STATUSES
    critique_result: CritiqueResult | None = None
    review_notes: str = ""
    review_result: ReviewResult | None = None
    sources: list[dict[str, str]] = Field(default_factory=list)

    def add_research_results(self, new_results: list[RetrievalResult]) -> None:
        """Append retrieval passes so loop-backs augment rather than replace.

        Duplicate *chunks* are de-duplicated later, by id, when the passes are
        merged for generation — here the raw passes are preserved in order.

        Args:
            new_results: Retrieval passes from the latest RESEARCH call.
        """
        self.research_results.extend(new_results)

    @property
    def current_answer(self) -> str:
        """The most recent answer — the revision if one exists."""
        return self.revised_answer or self.initial_answer

    @property
    def has_critique(self) -> bool:
        """Whether a structured critique has been recorded."""
        return self.critique_result is not None

    @property
    def has_review(self) -> bool:
        """Whether a structured self-check review has been recorded."""
        return self.review_result is not None


class SelfCritiqueResult(BaseModel):
    """Boundary object returned by the agent / graph after DELIVER."""

    query: str
    answer: str
    initial_answer: str
    revised: bool
    iterations: int
    critique: CritiqueResult | None = None
    sources: list[dict[str, str]] = Field(default_factory=list)
    status: str = "deliver"


class ReviewAgentResult(BaseModel):
    """Boundary object returned by :class:`ReviewAgent` after DELIVER.

    Carries the final answer plus the review metadata (verdict, iteration count,
    and whether a revision was applied) required by callers of the self-check
    pattern.
    """

    query: str
    answer: str
    initial_answer: str
    revised: bool
    iterations: int
    review: ReviewResult | None = None
    sources: list[dict[str, str]] = Field(default_factory=list)
    status: str = "deliver"
