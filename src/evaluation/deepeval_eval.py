"""DeepEval evaluation runner for RAG quality metrics.

Provides hallucination detection and other metrics using the DeepEval
library. Falls back to a deterministic lexical checker when deepeval is not
installed or when no LLM judge is configured, so evaluation stays runnable
offline. Evaluation never writes to production data (AGENTS.md invariant #6).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

from src.evaluation.datasets import EvalSample, sample_id_for

logger = structlog.get_logger(__name__)

# Thresholds handed to the real DeepEval LLM-judge metrics.
DEEPEVAL_THRESHOLDS: dict[str, float] = {
    "hallucination": 0.3,  # Max allowed hallucination score
    "answer_relevancy": 0.7,
    "faithfulness": 0.7,
}

# Lexical overlap is far coarser than an LLM judge, so the offline fallback uses
# relaxed thresholds. Applying the judge thresholds would reject well-grounded
# answers that merely paraphrase their context (ADR-004, Week 4).
LEXICAL_THRESHOLDS: dict[str, float] = {
    "hallucination": 0.35,
    "answer_relevancy": 0.6,
    "faithfulness": 0.4,
}

# Metrics where a lower score is better (thresholds are upper bounds).
_LOWER_IS_BETTER = frozenset({"hallucination"})

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class DeepEvalResult:
    """Result from DeepEval evaluation."""

    sample_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    passed: bool = True
    details: dict[str, str] = field(default_factory=dict)


def _tokenize(text: str) -> set[str]:
    """Lowercase ``text`` and split it into alphanumeric word tokens."""
    return set(_WORD_RE.findall(text.lower()))


def _lexical_scores(answer: str, query: str, contexts: list[str]) -> dict[str, float]:
    """Compute lexical overlap proxies for the three DeepEval metrics.

    Args:
        answer: The generated answer.
        query: The original query.
        contexts: The retrieved context chunks.

    Returns:
        Mapping of metric name to a score in ``[0, 1]``.
    """
    answer_tokens = _tokenize(answer)
    query_tokens = _tokenize(query)
    context_tokens = _tokenize(" ".join(contexts))
    grounding = context_tokens | query_tokens
    total = len(answer_tokens)
    hallucination = len(answer_tokens - grounding) / total if total else 1.0
    relevancy = len(answer_tokens & query_tokens) / len(query_tokens) if query_tokens else 0.0
    faithfulness = len(answer_tokens & context_tokens) / total if total else 0.0
    return {
        "hallucination": round(hallucination, 4),
        "answer_relevancy": round(relevancy, 4),
        "faithfulness": round(faithfulness, 4),
    }


def _passes_lexical(metrics: dict[str, float]) -> bool:
    """Check lexical metrics against the relaxed fallback thresholds."""
    for name, threshold in LEXICAL_THRESHOLDS.items():
        score = metrics.get(name, 0.0)
        if name in _LOWER_IS_BETTER:
            if score > threshold:
                return False
        elif score < threshold:
            return False
    return True


def _collect_metric_data(evaluation: object) -> list[tuple[str, float, bool, str]]:
    """Flatten a deepeval EvaluationResult into per-metric rows.

    Handles both the modern ``test_results[].metrics_data`` layout and the
    legacy ``test_results[].metrics`` layout so version drift degrades to the
    lexical fallback rather than an exception.

    Args:
        evaluation: Object returned by ``deepeval.evaluate``.

    Returns:
        Tuples of ``(metric_name, score, success, reason)``.
    """
    rows: list[tuple[str, float, bool, str]] = []
    test_results = getattr(evaluation, "test_results", None) or []
    for test_result in test_results:
        metric_data = getattr(test_result, "metrics_data", None) or getattr(
            test_result, "metrics", None
        )
        for metric in metric_data or []:
            name = str(getattr(metric, "name", "") or "unknown").lower().replace(" ", "_")
            score = float(getattr(metric, "score", 0.0) or 0.0)
            success = bool(getattr(metric, "success", True))
            reason = str(getattr(metric, "reason", "") or getattr(metric, "error", "") or "")
            rows.append((name, score, success, reason))
    return rows


class DeepEvalEvaluator:
    """Evaluator using DeepEval metrics with lexical fallback.

    When deepeval is installed and configured, uses real LLM-based metrics.
    Otherwise falls back to lexical overlap scoring.
    """

    def __init__(self, thresholds: dict[str, float] | None = None) -> None:
        """Initialize the evaluator.

        Args:
            thresholds: Custom metric thresholds. Defaults to DEEPEVAL_THRESHOLDS.
        """
        self._thresholds = thresholds or DEEPEVAL_THRESHOLDS
        self._deepeval_available = self._check_deepeval()

    def _check_deepeval(self) -> bool:
        """Check if deepeval is installed."""
        try:
            import deepeval  # noqa: F401 — import probe only
        except ImportError:
            logger.warning("deepeval not installed, using lexical fallback")
            return False
        return True

    async def evaluate(
        self,
        answer: str,
        query: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> DeepEvalResult:
        """Evaluate an answer using DeepEval metrics.

        Args:
            answer: The generated answer.
            query: The original query.
            contexts: The retrieved context chunks.
            ground_truth: Optional ground truth answer.

        Returns:
            DeepEvalResult with metrics and pass/fail status.
        """
        sample_id = sample_id_for(query)

        if self._deepeval_available:
            return await self._evaluate_with_deepeval(
                answer, query, contexts, ground_truth, sample_id
            )
        else:
            return await self._evaluate_lexical(answer, query, contexts, ground_truth, sample_id)

    async def _evaluate_with_deepeval(
        self,
        answer: str,
        query: str,
        contexts: list[str],
        ground_truth: str | None,
        sample_id: str,
    ) -> DeepEvalResult:
        """Evaluate using actual DeepEval metrics."""
        try:
            from deepeval.evaluate.evaluate import evaluate
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                BaseMetric,
                FaithfulnessMetric,
                HallucinationMetric,
            )
            from deepeval.test_case import LLMTestCase

            # HallucinationMetric reads `context`; the other metrics read
            # `retrieval_context`, so both are populated with the same chunks.
            test_case = LLMTestCase(
                input=query,
                actual_output=answer,
                expected_output=ground_truth,
                context=contexts or [""],
                retrieval_context=contexts or [""],
            )
            metrics: list[BaseMetric] = [
                FaithfulnessMetric(threshold=self._thresholds.get("faithfulness", 0.7)),
                AnswerRelevancyMetric(threshold=self._thresholds.get("answer_relevancy", 0.7)),
                HallucinationMetric(threshold=self._thresholds.get("hallucination", 0.3)),
            ]
            evaluation = evaluate([test_case], metrics)
            rows = _collect_metric_data(evaluation)
            if not rows:
                raise RuntimeError("deepeval returned no metric results")
        except Exception as e:  # noqa: BLE001 — fallback must never break evals
            logger.error("DeepEval evaluation failed, falling back to lexical", error=str(e))
            return await self._evaluate_lexical(answer, query, contexts, ground_truth, sample_id)

        return DeepEvalResult(
            sample_id=sample_id,
            metrics={name: score for name, score, _ok, _reason in rows},
            passed=all(ok for _name, _score, ok, _reason in rows),
            details={name: reason for name, _score, _ok, reason in rows},
        )

    async def _evaluate_lexical(
        self,
        answer: str,
        query: str,
        contexts: list[str],
        ground_truth: str | None,
        sample_id: str,
    ) -> DeepEvalResult:
        """Fallback lexical evaluation when deepeval is not available."""
        metrics = _lexical_scores(answer, query, contexts)
        passed = _passes_lexical(metrics)

        return DeepEvalResult(
            sample_id=sample_id,
            metrics=metrics,
            passed=passed,
            details={"method": "lexical_fallback"},
        )


async def run_deepeval_evaluation(
    samples: list[EvalSample],
    thresholds: dict[str, float] | None = None,
) -> list[DeepEvalResult]:
    """Run DeepEval evaluation on a list of samples.

    Args:
        samples: List of evaluation samples.
        thresholds: Custom metric thresholds.

    Returns:
        List of DeepEvalResult for each sample.
    """
    evaluator = DeepEvalEvaluator(thresholds)
    results = []

    for sample in samples:
        result = await evaluator.evaluate(
            answer=sample.answer,
            query=sample.question,
            contexts=sample.contexts,
            ground_truth=sample.ground_truth,
        )
        results.append(result)

        logger.info(
            "DeepEval sample evaluated",
            sample_id=result.sample_id,
            passed=result.passed,
            metrics=result.metrics,
        )

    return results
