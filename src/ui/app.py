"""DeepResearch AI — Streamlit entry point.

Run with:
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts only this script's directory on sys.path, not the project
# root, so `import src` fails under `streamlit run` (uvicorn works because it
# resolves via cwd). Bootstrap the root before any project import.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from src.ui.components.file_upload import upload_documents  # noqa: E402
from src.ui.pages import eval_dashboard, knowledge_base, research_chat  # noqa: E402

st.set_page_config(
    page_title="DeepResearch AI",
    page_icon="🔬",
    layout="wide",
)

with st.sidebar:
    st.title("🔬 DeepResearch AI")
    st.caption("Multi-agent research platform")

    st.divider()

    st.subheader("Quick Upload")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "md", "csv", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        upload_documents(uploaded_files)

    st.divider()

    page = st.radio(
        "Navigate",
        ["💬 Research Chat", "📚 Knowledge Base", "📊 Eval Dashboard"],
        label_visibility="collapsed",
    )

if page == "💬 Research Chat":
    research_chat.render()
elif page == "📚 Knowledge Base":
    knowledge_base.render()
elif page == "📊 Eval Dashboard":
    eval_dashboard.render()
