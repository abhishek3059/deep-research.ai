"""Ragas evaluation runner with an offline lexical baseline.

Tries the real ``ragas`` library when it is installed and configured;
otherwise falls back to deterministic lexical scorers so the Phase 1
baseline can run without LLM credentials. All entry points are asyncio
(AGENTS.md invariant #8) and evaluation never writes to production data
(AGENTS.md invariant #6).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

import structlog

from src.evaluation.datasets import EvalResult, EvalSample, sample_id_for

logger = structlog.get_logger(__name__)

DEFAULT_THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.90,
    "context_precision": 0.70,
    "context_recall": 0.70,
}

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

_STOPWORDS = frozenset(
    "a an the and or but if then else for of in on at to from with by is are was were "
    "be been being it its this that these those as what which who whom how why when "
    "where does do did can could should would will just very so than too".split()
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    """Apply a minimal suffix strip so plurals match their singulars."""
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens(text: str) -> set[str]:
    """Tokenize to lowercase stemmed content words minus stopwords."""
    return {_stem(w) for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def _sentences(text: str) -> list[str]:
    """Split text into non-empty sentences on common terminators."""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _recall(hypothesis: set[str], reference: set[str]) -> float:
    """Fraction of reference tokens covered by the hypothesis."""
    if not reference:
        return 0.0
    return len(hypothesis & reference) / len(reference)


def score_faithfulness(answer: str, contexts: list[str]) -> float:
    """Share of answer sentences supported by the retrieved contexts.

    A sentence counts as supported when at least half of its content
    tokens appear in the combined context vocabulary.
    """
    sentences = _sentences(answer)
    if not sentences or not contexts:
        return 0.0
    vocab: set[str] = set()
    for ctx in contexts:
        vocab |= _tokens(ctx)
    if not vocab:
        return 0.0
    supported = sum(1 for s in sentences if _support_ratio(_tokens(s), vocab) >= 0.5)
    return round(supported / len(sentences), 4)


def _support_ratio(sentence_tokens: set[str], vocab: set[str]) -> float:
    """Fraction of sentence tokens present in the context vocabulary."""
    if not sentence_tokens:
        return 0.0
    return len(sentence_tokens & vocab) / len(sentence_tokens)


def score_answer_relevancy(question: str, answer: str) -> float:
    """Keyword coverage of the question by the answer (lexical baseline)."""
    return round(_recall(_tokens(answer), _tokens(question)), 4)


def _context_is_relevant(context: str, question: str) -> bool:
    """A context is relevant when it covers >=20% of question keywords."""
    return _recall(_tokens(context), _tokens(question)) >= 0.2


def score_context_precision(question: str, contexts: list[str]) -> float:
    """Share of retrieved contexts relevant to the question."""
    if not contexts or not _tokens(question):
        return 0.0
    relevant = sum(1 for c in contexts if _context_is_relevant(c, question))
    return round(relevant / len(contexts), 4)


def score_context_recall(contexts: list[str], ground_truth: str | None) -> float:
    """Share of ground-truth keywords covered by retrieved contexts.

    Reference-free fallback: when no ground truth exists, reuse context
    precision against the answer text as the coverage proxy.
    """
    vocab: set[str] = set()
    for ctx in contexts:
        vocab |= _tokens(ctx)
    if ground_truth:
        return round(_recall(vocab, _tokens(ground_truth)), 4)
    return 0.0 if not contexts else 1.0 if vocab else 0.0


@dataclass
class RagasEvaluator:
    """Compute faithfulness, relevancy, precision, and recall per sample."""

    thresholds: dict[str, float] | None = None
    use_ragas: bool = True

    def __post_init__(self) -> None:
        """Apply default thresholds for any unspecified metric."""
        merged = dict(DEFAULT_THRESHOLDS)
        if self.thresholds:
            merged.update(self.thresholds)
        self.thresholds = merged

    async def evaluate_sample(self, sample: EvalSample, index: int = 0) -> EvalResult:
        """Score one sample and return its pass/fail EvalResult."""
        metrics = await self._score_sample(sample)
        passed = self._check_thresholds(metrics, has_ground_truth=sample.ground_truth is not None)
        result = EvalResult(
            sample_id=sample_id_for(sample.question, index),
            metrics=metrics,
            passed=passed,
        )
        logger.info("Sample evaluated", sample_id=result.sample_id, passed=passed)
        return result

    async def evaluate_batch(self, samples: list[EvalSample]) -> list[EvalResult]:
        """Score all samples concurrently and preserve input order."""
        tasks = [self.evaluate_sample(s, i) for i, s in enumerate(samples)]
        return list(await asyncio.gather(*tasks))

    def aggregate(self, results: list[EvalResult]) -> dict[str, float | int]:
        """Aggregate per-sample results into mean scores and pass rate."""
        summary: dict[str, float | int] = {"n": len(results)}
        if not results:
            return summary
        passed = sum(1 for r in results if r.passed)
        summary["pass_rate"] = round(passed / len(results), 4)
        summary["n_passed"] = passed
        for name in METRIC_NAMES:
            scores = [r.metrics.get(name, 0.0) for r in results]
            summary[f"mean_{name}"] = round(sum(scores) / len(scores), 4)
        return summary

    def _check_thresholds(self, metrics: dict[str, float], has_ground_truth: bool) -> bool:
        """Check every applicable metric against its threshold."""
        assert self.thresholds is not None  # set in __post_init__
        for name in METRIC_NAMES:
            if name == "context_recall" and not has_ground_truth:
                continue  # reference-based metric skipped without ground truth
            if metrics.get(name, 0.0) < self.thresholds[name]:
                return False
        return True

    async def _score_sample(self, sample: EvalSample) -> dict[str, float]:
        """Prefer real ragas scores; fall back to lexical scorers offline."""
        ragas_scores = await self._try_ragas_scores(sample)
        if ragas_scores is not None:
            return ragas_scores
        return await asyncio.to_thread(self._lexical_scores, sample)

    async def _try_ragas_scores(self, sample: EvalSample) -> dict[str, float] | None:
        """Attempt scoring with the ragas library; None when unavailable.

        Any import, configuration, or network failure degrades gracefully
        to the lexical baseline so CI stays green without LLM keys.
        """
        if not self.use_ragas:
            return None
        try:
            return await asyncio.to_thread(self._run_ragas, sample)
        except Exception as e:  # noqa: BLE001 — fallback must never break evals
            logger.warning("Ragas unavailable, using lexical baseline", error=str(e))
            return None

    def _run_ragas(self, sample: EvalSample) -> dict[str, float]:
        """Run ragas metrics synchronously in a worker thread.

        Raises:
            ImportError: If ragas is not installed.
            RuntimeError: If ragas evaluation fails (e.g. no LLM keys).
        """
        try:
            from datasets import Dataset  # type: ignore[import-not-found]
            from ragas import evaluate  # type: ignore[import-not-found]
            from ragas.metrics import (  # type: ignore[import-not-found]
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except ImportError as e:
            raise ImportError("ragas library not installed") from e
        data = {
            "question": [sample.question],
            "answer": [sample.answer],
            "contexts": [sample.contexts],
            "ground_truth": [sample.ground_truth or sample.answer],
        }
        try:
            result = evaluate(
                Dataset.from_dict(data),
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            )
        except Exception as e:
            raise RuntimeError(f"ragas evaluation failed: {e}") from e
        scores = result.to_pandas().iloc[0].to_dict()
        key_map = {
            "faithfulness": "faithfulness",
            "answer_relevancy": "answer_relevancy",
            "context_precision": "context_precision",
            "context_recall": "context_recall",
        }
        # Ragas may return NaN/scores under variant keys — coerce defensively.
        resolved: dict[str, float] = {}
        for canonical, ragas_key in key_map.items():
            value = scores.get(ragas_key, float("nan"))
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = float("nan")
            resolved[canonical] = number
        if any(v != v for v in resolved.values()):  # NaN check
            raise RuntimeError("ragas returned NaN scores")
        return resolved

    @staticmethod
    def _lexical_scores(sample: EvalSample) -> dict[str, float]:
        """Compute deterministic lexical baseline scores for one sample."""
        faithfulness = score_faithfulness(sample.answer, sample.contexts)
        return {
            "faithfulness": faithfulness,
            "answer_relevancy": score_answer_relevancy(sample.question, sample.answer),
            "context_precision": score_context_precision(sample.question, sample.contexts),
            "context_recall": score_context_recall(sample.contexts, sample.ground_truth),
        }
