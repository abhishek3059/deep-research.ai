"""Evaluation dashboard page: browse and preview eval HTML reports."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

REPORT_DIR = Path("data/eval_reports")

METRIC_TARGETS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.90,
    "context_precision": 0.70,
    "context_recall": 0.70,
}


def _list_reports() -> list[Path]:
    """Return eval reports newest-first; empty when none exist yet."""
    if not REPORT_DIR.is_dir():
        return []
    return sorted(
        REPORT_DIR.glob("eval_report-*.html"), key=lambda p: p.stat().st_mtime, reverse=True
    )


def render() -> None:
    """Render the evaluation dashboard page."""
    st.title("Evaluation Dashboard")

    st.subheader("Production metric bars")
    st.table(
        [
            {"metric": name, "pass_threshold": f">= {bar:.2f}"}
            for name, bar in METRIC_TARGETS.items()
        ]
    )
    st.caption("Scores come from Ragas/DeepEval judges, or the lexical fallback offline.")

    st.divider()
    st.subheader("Reports")

    reports = _list_reports()
    if not reports:
        st.info(
            "No reports yet. Run `uv run python scripts/demo.py` or "
            "`uv run python scripts/run_evals.py` to generate one."
        )
        return

    options = {p.name: p for p in reports}
    choice = st.selectbox("Report", list(options), index=0)
    if choice is None:
        return
    path = options[choice]
    st.caption(f"{path.stat().st_size / 1024:.1f} KB")

    with path.open(encoding="utf-8") as fh:
        components.html(fh.read(), height=700, scrolling=True)

    with path.open("rb") as fh:
        st.download_button("Download HTML", fh, file_name=path.name, mime="text/html")
