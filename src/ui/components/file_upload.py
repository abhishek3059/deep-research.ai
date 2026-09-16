"""File upload component for the knowledge base."""

from __future__ import annotations

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

ALLOWED_EXTENSIONS = {"pdf", "md", "csv", "txt"}


def _validate_extension(filename: str) -> bool:
    """Check whether the file extension is supported."""
    ext = filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_EXTENSIONS


def upload_documents(uploaded_files: list[object]) -> None:
    """Upload selected documents to the ingestion API.

    Args:
        uploaded_files: List of UploadedFile objects from st.file_uploader.
    """
    if not uploaded_files:
        return

    success_count = 0
    error_count = 0

    for file in uploaded_files:
        filename = getattr(file, "name", "unknown")
        if not _validate_extension(filename):
            st.warning(f"Skipping unsupported file type: {filename}")
            error_count += 1
            continue

        with st.spinner(f"Ingesting `{filename}`..."):
            try:
                files = {"file": (filename, file.getvalue(), "application/octet-stream")}
                response = httpx.post(
                    f"{API_BASE}/api/v1/ingest",
                    files=files,
                    timeout=120.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"`{filename}` ingested ({data.get('chunk_count', 0)} chunks)")
                    success_count += 1
                else:
                    st.error(f"Failed to ingest `{filename}`: HTTP {response.status_code}")
                    error_count += 1
            except httpx.ConnectError:
                st.error("Cannot connect to API server. Is it running on port 8000?")
                error_count += 1
                break
            except Exception as e:
                st.error(f"Error ingesting `{filename}`: {e}")
                error_count += 1

    if success_count:
        st.toast(f"Successfully ingested {success_count} document(s)", icon="✅")
    if error_count:
        st.toast(f"{error_count} file(s) failed", icon="⚠️")
