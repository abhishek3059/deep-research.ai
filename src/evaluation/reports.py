"""HTML report generation for evaluation runs.

Reports are written under ``data/eval_reports/`` and contain aggregate
metric means, pass/fail status, and a per-sample breakdown. Generating a
report never modifies production data.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime
from pathlib import Path

import structlog

from src.evaluation.datasets import EvalResult

logger = structlog.get_logger(__name__)

DEFAULT_REPORT_DIR = Path("data/eval_reports")
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def summarize_results(results: list[EvalResult]) -> dict[str, float | int]:
    """Aggregate per-sample results into means and pass counts."""
    summary: dict[str, float | int] = {"n": len(results)}
    if not results:
        return summary
    passed = sum(1 for r in results if r.passed)
    summary["n_passed"] = passed
    summary["pass_rate"] = round(passed / len(results), 4)
    for name in METRIC_NAMES:
        scores = [r.metrics.get(name, 0.0) for r in results]
        summary[f"mean_{name}"] = round(sum(scores) / len(scores), 4)
    return summary


def render_html_report(
    results: list[EvalResult],
    summary: dict[str, float | int] | None = None,
    title: str = "DeepResearch AI — Evaluation Report",
) -> str:
    """Render results as a standalone HTML document string."""
    summary = summary if summary is not None else summarize_results(results)
    generated_at = datetime.now(UTC).isoformat()

    mean_cells = "".join(
        f"<div class='stat'><span class='label'>mean {html.escape(name)}</span>"
        f"<span class='value'>{summary.get(f'mean_{name}', 'n/a')}</span></div>"
        for name in METRIC_NAMES
    )
    rows = "\n".join(_sample_row(r) for r in results) or (
        "<tr><td colspan='7'>No samples evaluated.</td></tr>"
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
.stats {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }}
.stat {{ border: 1px solid #ddd; border-radius: 8px; padding: 0.75rem 1rem; }}
.stat .label {{ display: block; font-size: 0.8rem; color: #555; }}
.stat .value {{ font-size: 1.4rem; font-weight: bold; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
th, td {{ border: 1px solid #ddd; padding: 0.5rem; text-align: left; font-size: 0.9rem; }}
th {{ background: #f5f5f5; }}
.pass {{ color: #0a7d2c; font-weight: bold; }}
.fail {{ color: #b3261e; font-weight: bold; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p>Generated at {html.escape(generated_at)} &middot; Samples: {summary.get("n", 0)}
&middot; Passed: {summary.get("n_passed", 0)}
&middot; Pass rate: {summary.get("pass_rate", "n/a")}</p>
<div class="stats">{mean_cells}</div>
<h2>Sample breakdown</h2>
<table>
<tr><th>Sample</th><th>Faithfulness</th><th>Relevancy</th>
<th>Ctx precision</th><th>Ctx recall</th><th>Status</th><th>Evaluated</th></tr>
{rows}
</table>
</body>
</html>
"""


def _sample_row(result: EvalResult) -> str:
    """Render one per-sample table row."""
    status = "<span class='pass'>PASS</span>" if result.passed else "<span class='fail'>FAIL</span>"
    cells = "".join(f"<td>{result.metrics.get(name, 0.0):.3f}</td>" for name in METRIC_NAMES)
    return (
        f"<tr><td>{html.escape(result.sample_id)}</td>{cells}"
        f"<td>{status}</td><td>{html.escape(result.timestamp)}</td></tr>"
    )


def save_html_report(
    results: list[EvalResult],
    output_dir: Path | str = DEFAULT_REPORT_DIR,
    title: str = "DeepResearch AI — Evaluation Report",
) -> Path:
    """Render results and save the HTML report to a timestamped file.

    Args:
        results: Per-sample evaluation results.
        output_dir: Directory for report files (created on demand).
        title: Report heading.

    Returns:
        Path of the written HTML file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"eval_report-{stamp}.html"
    summary = summarize_results(results)
    path.write_text(render_html_report(results, summary, title), encoding="utf-8")
    logger.info("Evaluation report saved", path=str(path), samples=len(results))
    return path
