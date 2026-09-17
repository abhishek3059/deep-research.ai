"""Public API for the evaluation module."""

from src.evaluation.datasets import (
    EvalResult,
    EvalSample,
    GoldenDatasetManager,
    sample_id_for,
)
from src.evaluation.deepeval_eval import (
    DeepEvalEvaluator,
    DeepEvalResult,
    run_deepeval_evaluation,
)
from src.evaluation.ragas_eval import DEFAULT_THRESHOLDS, RagasEvaluator
from src.evaluation.reports import (
    render_html_report,
    save_html_report,
    summarize_results,
)

__all__ = [
    "DEFAULT_THRESHOLDS",
    "DeepEvalEvaluator",
    "DeepEvalResult",
    "EvalResult",
    "EvalSample",
    "GoldenDatasetManager",
    "RagasEvaluator",
    "render_html_report",
    "run_deepeval_evaluation",
    "sample_id_for",
    "save_html_report",
    "summarize_results",
]
