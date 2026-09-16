"""Knowledge base management page."""

from __future__ import annotations

import httpx
import streamlit as st

from src.ui.components.file_upload import upload_documents

API_BASE = "http://localhost:8000"


def _fetch_documents() -> list[dict[str, str]]:
    """Fetch ingested documents from the API."""
    try:
        response = httpx.get(f"{API_BASE}/api/v1/health", timeout=10.0)
        if response.status_code == 200:
            return [{"name": "Knowledge base is active", "status": "ok"}]
    except httpx.ConnectError:
        return [{"name": "API server not reachable", "status": "error"}]
    return []


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

    st.subheader("Ingested Documents")

    health = _fetch_documents()
    if health and health[0].get("status") == "error":
        st.error(health[0]["name"])
        st.info(
            "Start the API server first: `uv run python -m src.api.main`"
        )
        return

    st.success("API server is running")

    if st.button("Refresh", type="secondary"):
        st.rerun()

    st.info(
        "Document listing will be available once the vector store "
        "implements a list/collection info endpoint."
    )


if __name__ == "__main__":
    render()
