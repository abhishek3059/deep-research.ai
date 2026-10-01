"""Load and ingest sample documents into the vector store.

Usage:
    uv run python scripts/ingest_sample_data.py [--persist-dir DIR]

Writes three starter Markdown fixtures into ``data/sample_docs/`` (if missing),
ingests them through :class:`~src.ingestion.pipeline.IngestionPipeline`, and
upserts the resulting chunks into ChromaDB via
:func:`~src.vectorstore.manager.get_store`.

Requires an embedding API key (``OPENAI_API_KEY``) — the default
:class:`~src.ingestion.embedder.Embedder` calls OpenAI. Exits with a clear
message when the key is absent; use ``scripts/demo.py`` for a key-free run.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure the project root is importable when run as a script file.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog

logger = structlog.get_logger(__name__)

SAMPLE_DIR = Path("data/sample_docs")

SAMPLE_DOCS: dict[str, str] = {
    "rag_overview.md": """# Retrieval-Augmented Generation at DeepResearch AI

Retrieval-Augmented Generation (RAG) grounds large language model answers in
retrieved evidence. Each document is split into overlapping chunks of 512
tokens with a 64-token overlap, so key facts stay near the top of each chunk.
DeepResearch AI implements RAG as a five-stage pipeline: document ingestion,
persistent vector storage, hybrid retrieval, grounded generation, and quality
evaluation.

Documents are loaded from PDF, web, Markdown, CSV, and text sources.
Duplicate chunks are removed by content hash before embedding.

The vector store is ChromaDB with a persistent client, so ingested knowledge
survives restarts. Switching providers requires only a configuration change.
""",
    "hybrid_retrieval.md": """# Hybrid Retrieval with Reciprocal Rank Fusion

DeepResearch AI combines dense vector search with BM25 sparse keyword search.
Dense retrieval captures semantic similarity through embeddings, while BM25
captures exact term matches that embeddings sometimes miss.

The two ranked lists are fused with Reciprocal Rank Fusion (RRF). Each
document scores sum(1 / (k + rank)) with k=60, a value chosen from evaluation
runs. Fusion needs no score normalization between the dense and sparse legs.

After fusion, a cross-encoder reranker scores query-document pairs and keeps
the top results. Multi-query expansion adds alternative phrasings of the user
question to improve recall before retrieval runs.
""",
    "guardrails.md": """# Guardrails and Evaluation

Every answer passes through three guards. The input guard rejects prompt
injection, off-topic questions, and over-long queries. The output guard checks
response format, citation presence, and content safety. The hallucination guard
scores how well each claim is grounded in the retrieved context and rejects
answers below threshold.

Quality is measured with Ragas metrics: faithfulness above 0.85 and answer
relevancy above 0.90. DeepEval provides an independent hallucination metric.
Evaluation reads production data but never modifies it, and every run saves a
timestamped HTML report under data/eval_reports/.
""",
}


def ensure_sample_docs(sample_dir: Path = SAMPLE_DIR) -> list[Path]:
    """Write starter fixtures if missing; return all Markdown paths."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    for name, content in SAMPLE_DOCS.items():
        path = sample_dir / name
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            logger.info("Wrote sample document", path=str(path))
    return sorted(sample_dir.glob("*.md"))


async def ingest_all(sample_dir: Path = SAMPLE_DIR) -> int:
    """Ingest every Markdown fixture and upsert chunks; return chunk count."""
    from src.config.settings import settings
    from src.ingestion.pipeline import IngestionPipeline
    from src.vectorstore.manager import get_store

    if not settings.openai_api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add it to .env for real ingestion, "
            "or run 'uv run python scripts/demo.py' for a key-free demo."
        )

    paths = ensure_sample_docs(sample_dir)
    if not paths:
        raise SystemExit(f"No Markdown files found in {sample_dir}")

    pipeline = IngestionPipeline()
    store = get_store()
    total = 0
    for path in paths:
        chunks = await pipeline.ingest(str(path))
        stored = await store.upsert(chunks)
        total += stored
        logger.info("Ingested", source=str(path), chunks=stored)
    return total


async def main(persist_info: bool = True) -> None:
    """Entry point: ingest fixtures and report the chunk count."""
    from src.config.settings import settings

    total = await ingest_all()
    if persist_info:
        print(f"Ingested {total} chunks into {settings.chroma_persist_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest sample documents.")
    parser.parse_args()
    asyncio.run(main())
