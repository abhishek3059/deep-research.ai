"""Knowledge base management page."""

from __future__ import annotations

from typing import Any, cast

import httpx
import streamlit as st

from src.ui.components.file_upload import upload_documents
from src.ui.config import API_BASE, HEALTH_TIMEOUT_SECONDS


def _fetch_health() -> dict[str, Any] | None:
    """Return the API health payload, or None when unreachable."""
    try:
        response = httpx.get(f"{API_BASE}/api/v1/health", timeout=HEALTH_TIMEOUT_SECONDS)
        if response.status_code == 200:
            return cast("dict[str, Any]", response.json())
    except httpx.HTTPError:
        return None
    return None


def render() -> None:
    """Render the knowledge base management page."""
    st.title("📚 Knowledge Base")

    st.subheader("Upload Documents")
    st.caption("Supported formats: PDF, Markdown, CSV, TXT")

    uploaded_files = st.file_uploader(
        "Choose files to ingest",
        type=["pdf", "md", "csv", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        upload_documents(uploaded_files)

    st.divider()

    st.subheader("Vector Store")

    if st.button("Refresh", type="secondary"):
        st.rerun()

    health = _fetch_health()
    if health is None:
        st.error(f"API server not reachable at {API_BASE}")
        st.info("Start it with `make api`, or rebuild the container.")
        return

    chunks = health.get("chunks")
    backend = health.get("vector_store", "unknown")

    if isinstance(chunks, int) and chunks > 0:
        st.metric("Chunks indexed", chunks)
        st.caption(f"Backend: {backend} · embeddings persist across restarts")
    elif isinstance(chunks, int):
        st.warning("The vector store is empty — no documents have been ingested yet.")
        st.info("Upload a document above, or run `make demo` to load sample data.")
    else:
        st.info("Backend health OK. Chunk count is not reported by this store.")


if __name__ == "__main__":
    render()
