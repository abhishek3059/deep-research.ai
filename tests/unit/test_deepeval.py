"""Tests for DeepEval evaluation runner."""

from __future__ import annotations

from typing import Any

import pytest

from src.evaluation.datasets import EvalSample
from src.evaluation.deepeval_eval import (
    DeepEvalEvaluator,
    DeepEvalResult,
    run_deepeval_evaluation,
)


@pytest.fixture(autouse=True)
def _force_lexical_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin evaluation to the offline lexical path so tests stay deterministic.

    The real DeepEval metrics require an LLM judge (and API credentials); CI
    only exercises the fallback, matching the Ragas baseline strategy.
    """
    monkeypatch.setattr(DeepEvalEvaluator, "_check_deepeval", lambda self: False)


class TestDeepEvalEvaluator:
    """Tests for the DeepEvalEvaluator class."""

    @pytest.fixture
    def evaluator(self) -> DeepEvalEvaluator:
        """Create a fresh evaluator for each test."""
        return DeepEvalEvaluator()

    @pytest.mark.asyncio
    async def test_evaluate_with_good_answer(self, evaluator: DeepEvalEvaluator) -> None:
        """Test evaluation with a well-grounded answer."""
        result = await evaluator.evaluate(
            answer="The capital of France is Paris. It is located in northern France.",
            query="What is the capital of France?",
            contexts=["France is a country in Europe. Its capital is Paris."],
        )
        assert isinstance(result, DeepEvalResult)
        assert result.passed is True
        assert "hallucination" in result.metrics
        assert "answer_relevancy" in result.metrics

    @pytest.mark.asyncio
    async def test_evaluate_with_hallucinated_answer(self, evaluator: DeepEvalEvaluator) -> None:
        """Test evaluation with a hallucinated answer."""
        result = await evaluator.evaluate(
            answer="The capital of France is London. It is in England.",
            query="What is the capital of France?",
            contexts=["France is a country in Europe. Its capital is Paris."],
        )
        assert isinstance(result, DeepEvalResult)
        # Hallucination score should be high (bad)
        assert result.metrics.get("hallucination", 0) > 0.3

    @pytest.mark.asyncio
    async def test_evaluate_with_empty_context(self, evaluator: DeepEvalEvaluator) -> None:
        """Test evaluation with empty context."""
        result = await evaluator.evaluate(
            answer="Some answer about a topic.",
            query="What is the topic?",
            contexts=[],
        )
        assert isinstance(result, DeepEvalResult)
        assert result.sample_id is not None

    @pytest.mark.asyncio
    async def test_evaluate_uses_lexical_fallback(self, evaluator: DeepEvalEvaluator) -> None:
        """The offline path is flagged in the result details."""
        result = await evaluator.evaluate(
            answer="Python is a programming language.",
            query="What is Python?",
            contexts=["Python is a high-level programming language."],
        )
        assert result.details["method"] == "lexical_fallback"

    @pytest.mark.asyncio
    async def test_evaluate_falls_back_when_deepeval_raises(
        self, evaluator: DeepEvalEvaluator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing LLM judge degrades to the lexical fallback, never raises."""
        import deepeval

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("LLM judge unavailable")

        monkeypatch.setattr(deepeval, "evaluate", _boom)
        evaluator._deepeval_available = True

        result = await evaluator.evaluate(
            answer="Python is a programming language.",
            query="What is Python?",
            contexts=["Python is a high-level programming language."],
        )
        assert result.details["method"] == "lexical_fallback"


class TestRunDeepEvalEvaluation:
    """Tests for the run_deepeval_evaluation function."""

    @pytest.mark.asyncio
    async def test_run_evaluation_on_samples(self) -> None:
        """Test running evaluation on multiple samples."""
        samples = [
            EvalSample(
                question="What is Python?",
                answer="Python is a programming language.",
                contexts=["Python is a high-level programming language."],
                ground_truth="Python is a programming language.",
            ),
            EvalSample(
                question="What is Java?",
                answer="Java is a coffee beverage.",
                contexts=["Java is a programming language."],
                ground_truth="Java is a programming language.",
            ),
        ]
        results = await run_deepeval_evaluation(samples)
        assert len(results) == 2
        assert all(isinstance(r, DeepEvalResult) for r in results)
