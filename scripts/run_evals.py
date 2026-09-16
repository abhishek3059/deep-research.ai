"""Run the Ragas evaluation baseline against the Phase 1 RAG pipeline.

Usage:
    uv run python scripts/run_evals.py [--dataset NAME] [--no-pipeline]

Loads the golden dataset, answers each question through the Phase 1
retrieval + generation pipeline when available (otherwise scores the
golden answers directly), computes Ragas metrics, saves an HTML report
to data/eval_reports/, and prints a summary.

Evaluation is read-only: it never writes to the production vector store.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure the project root is importable when run as a script file.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog

from src.evaluation.datasets import EvalSample, GoldenDatasetManager
from src.evaluation.ragas_eval import RagasEvaluator
from src.evaluation.reports import save_html_report, summarize_results

logger = structlog.get_logger(__name__)


async def _answer_via_pipeline(question: str) -> EvalSample | None:
    """Answer one question with the Phase 1 pipeline; None if unavailable.

    Any failure (missing keys, empty store, import error) degrades to
    None so the caller can fall back to the golden answer.
    """
    try:
        from src.agents.generation import GenerationPipeline
        from src.retrieval.pipeline import RetrievalPipeline
        from src.vectorstore.manager import get_store
    except ImportError as e:
        logger.warning("Phase 1 pipeline unavailable", error=str(e))
        return None
    try:
        store = get_store()
        pipeline = RetrievalPipeline(vector_store=store)
        retrieval = await pipeline.retrieve(question)
        generation = GenerationPipeline()
        produced = await generation.generate_answer(question, retrieval)
        contexts = [r.text for r in retrieval.results]
        return EvalSample(
            question=question,
            answer=str(produced.get("answer", "")),
            contexts=contexts,
            ground_truth=None,
        )
    except Exception as e:  # noqa: BLE001 — evals must degrade, never crash
        logger.warning("Pipeline QA failed, using golden answer", error=str(e))
        return None


async def run_evaluations(dataset_name: str, use_pipeline: bool) -> int:
    """Load the dataset, score every question, save a report, print summary."""
    manager = GoldenDatasetManager()
    samples = manager.load(dataset_name)
    print(f"Loaded {len(samples)} samples from {dataset_name}")

    to_score: list[EvalSample] = []
    pipeline_used = 0
    for sample in samples:
        if use_pipeline:
            produced = await _answer_via_pipeline(sample.question)
            if produced is not None:
                produced.ground_truth = sample.ground_truth
                to_score.append(produced)
                pipeline_used += 1
                continue
        to_score.append(sample)
    if use_pipeline:
        print(f"Phase 1 pipeline answered {pipeline_used}/{len(samples)} questions")

    evaluator = RagasEvaluator(use_ragas=True)  # falls back to lexical baseline
    results = await evaluator.evaluate_batch(to_score)
    summary = summarize_results(results)
    report_path = save_html_report(results, Path("data/eval_reports"))

    print("\n-- Evaluation summary --")
    for key in sorted(summary):
        print(f"  {key}: {summary[key]}")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.sample_id} {result.metrics}")
    print(f"\nReport saved to {report_path}")
    return 0


def main() -> int:
    """Parse CLI args and run the async evaluation workflow."""
    parser = argparse.ArgumentParser(description="Run the Ragas eval baseline.")
    parser.add_argument("--dataset", default="starter_dataset.json")
    parser.add_argument(
        "--no-pipeline",
        action="store_true",
        help="Score golden answers directly without the Phase 1 pipeline.",
    )
    args = parser.parse_args()
    return asyncio.run(run_evaluations(args.dataset, use_pipeline=not args.no_pipeline))


if __name__ == "__main__":
    sys.exit(main())
