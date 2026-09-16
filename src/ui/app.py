"""DeepResearch AI — Streamlit entry point.

Run with:
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import streamlit as st

from src.ui.components.file_upload import upload_documents
from src.ui.pages import knowledge_base, research_chat

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
        ["💬 Research Chat", "📚 Knowledge Base"],
        label_visibility="collapsed",
    )

if page == "💬 Research Chat":
    research_chat.render()
elif page == "📚 Knowledge Base":
    knowledge_base.render()
