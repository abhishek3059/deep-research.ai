"""Unit tests for the Ragas evaluation runner and HTML reports."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.datasets import EvalSample
from src.evaluation.ragas_eval import (
    DEFAULT_THRESHOLDS,
    RagasEvaluator,
    score_answer_relevancy,
    score_context_precision,
    score_context_recall,
    score_faithfulness,
)
from src.evaluation.reports import render_html_report, save_html_report, summarize_results


def _grounded_sample() -> EvalSample:
    return EvalSample(
        question="What is reciprocal rank fusion?",
        answer="Reciprocal rank fusion merges ranked lists.",
        contexts=["Reciprocal rank fusion merges ranked lists from dense and sparse retrieval."],
        ground_truth="Reciprocal rank fusion merges lists",
    )


def test_evaluate_grounded_answer_passes() -> None:
    async def _run() -> None:
        evaluator = RagasEvaluator(use_ragas=False)
        result = await evaluator.evaluate_sample(_grounded_sample())
        assert result.passed is True
        assert result.metrics["faithfulness"] == 1.0
        assert result.metrics["answer_relevancy"] == 1.0
        assert result.sample_id.startswith("sample-")

    _await(_run())


def test_evaluate_hallucinated_answer_fails() -> None:
    async def _run() -> None:
        sample = EvalSample(
            question="What is reciprocal rank fusion?",
            answer="Penguins migrate to the Sahara desert every winter.",
            contexts=["Reciprocal rank fusion merges ranked lists."],
            ground_truth="Reciprocal rank fusion merges lists",
        )
        result = await RagasEvaluator(use_ragas=False).evaluate_sample(sample)
        assert result.passed is False
        assert result.metrics["faithfulness"] == 0.0
        assert result.metrics["answer_relevancy"] == 0.0

    _await(_run())


def test_evaluate_empty_contexts_scores_zero_and_fails() -> None:
    async def _run() -> None:
        sample = EvalSample(
            question="What is reciprocal rank fusion?",
            answer="Reciprocal rank fusion merges ranked lists.",
            contexts=[],
            ground_truth="Reciprocal rank fusion merges lists",
        )
        result = await RagasEvaluator(use_ragas=False).evaluate_sample(sample)
        assert result.metrics["faithfulness"] == 0.0
        assert result.metrics["context_precision"] == 0.0
        assert result.passed is False

    _await(_run())


def test_evaluate_without_ground_truth_skips_context_recall() -> None:
    async def _run() -> None:
        sample = _grounded_sample()
        sample.ground_truth = None
        result = await RagasEvaluator(use_ragas=False).evaluate_sample(sample)
        assert result.passed is True  # reference-based recall is skipped

    _await(_run())


def test_evaluate_batch_preserves_order() -> None:
    async def _run() -> None:
        samples = [_grounded_sample(), _grounded_sample()]
        results = await RagasEvaluator(use_ragas=False).evaluate_batch(samples)
        assert len(results) == 2
        assert results[0].sample_id != results[1].sample_id  # index disambiguates
        assert all(r.passed for r in results)

    _await(_run())


def test_custom_thresholds_change_verdict() -> None:
    async def _run() -> None:
        strict = RagasEvaluator(
            thresholds={"faithfulness": 1.1, "answer_relevancy": 0.9},
            use_ragas=False,
        )
        result = await strict.evaluate_sample(_grounded_sample())
        assert result.passed is False
        assert strict.thresholds is not None
        assert strict.thresholds["context_precision"] == DEFAULT_THRESHOLDS["context_precision"]

    _await(_run())


def test_aggregate_computes_means_and_pass_rate() -> None:
    async def _run() -> None:
        evaluator = RagasEvaluator(use_ragas=False)
        results = await evaluator.evaluate_batch([_grounded_sample(), _grounded_sample()])
        summary = evaluator.aggregate(results)
        assert summary["n"] == 2
        assert summary["pass_rate"] == 1.0
        assert summary["mean_faithfulness"] == 1.0

    _await(_run())


def test_aggregate_with_no_results_returns_count_only() -> None:
    assert RagasEvaluator(use_ragas=False).aggregate([]) == {"n": 0}


def test_default_thresholds_match_contract_targets() -> None:
    assert DEFAULT_THRESHOLDS["faithfulness"] == 0.85
    assert DEFAULT_THRESHOLDS["answer_relevancy"] == 0.90


def test_lexical_scorers_handle_edge_inputs() -> None:
    assert score_faithfulness("", []) == 0.0
    assert score_faithfulness("Some answer.", []) == 0.0
    assert score_answer_relevancy("", "An answer.") == 0.0
    assert score_context_precision("A question?", []) == 0.0
    assert score_context_recall([], "truth") == 0.0
    assert 0.0 <= score_context_precision("What is RRF?", ["RRF merges lists."]) <= 1.0
    assert score_context_recall(["RRF merges ranked lists."], "RRF merges lists") == 1.0


def test_render_report_contains_status_and_scores() -> None:
    async def _run() -> None:
        results = await RagasEvaluator(use_ragas=False).evaluate_batch([_grounded_sample()])
        page = render_html_report(results)
        assert "<html" in page
        assert "PASS" in page
        assert results[0].sample_id in page

    _await(_run())


def test_save_report_writes_timestamped_html(tmp_path: Path) -> None:
    async def _run() -> None:
        results = await RagasEvaluator(use_ragas=False).evaluate_batch([_grounded_sample()])
        path = save_html_report(results, tmp_path)
        assert path.suffix == ".html"
        assert path.is_file()
        assert "PASS" in path.read_text(encoding="utf-8")
        summary = summarize_results(results)
        assert summary["n"] == 1

    _await(_run())


def _await(coro: object) -> None:
    """Drive an async test body without depending on pytest-asyncio config."""
    import asyncio

    assert asyncio.iscoroutine(coro)
    asyncio.run(coro)  # type: ignore[arg-type]
