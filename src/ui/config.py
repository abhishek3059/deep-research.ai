"""UI wiring — single source for API endpoint and timeouts.

Kept separate from :mod:`src.config.settings` because the UI is a thin HTTP
client: it talks to the API over the network and must not construct model
clients or touch the vector store directly.
"""

from __future__ import annotations

import os
from pathlib import Path

# Resolved from this file, not cwd: Streamlit does not guarantee the working
# directory, and under Docker the reports live on the mounted data volume.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_REPORT_DIR = PROJECT_ROOT / "data" / "eval_reports"

API_BASE = os.environ.get("API_BASE", "http://localhost:8000").rstrip("/")

# The full agent pipeline (retrieve -> generate -> critique -> deliver) issues
# several sequential LLM calls. Measured end-to-end latency is ~270s, so a
# 120s budget made every chat request time out on the client while the server
# was still working. Default is generous to absorb provider rate limiting.
QUERY_TIMEOUT_SECONDS = float(os.environ.get("QUERY_TIMEOUT_SECONDS", "600"))

# Ingestion is a single embedding pass over the document, not an LLM loop.
INGEST_TIMEOUT_SECONDS = float(os.environ.get("INGEST_TIMEOUT_SECONDS", "180"))

HEALTH_TIMEOUT_SECONDS = 15.0
