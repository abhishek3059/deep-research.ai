"""End-to-end demo of the DeepResearch AI pipeline (key-free by default).

Usage:
    uv run python scripts/demo.py [--report-dir DIR]

Runs ingest -> retrieve -> guard -> evaluate over three starter fixtures
using deterministic hash-bucket embeddings, so the full machinery works with
no API keys: ChromaDB round-trip, BM25 + dense + RRF fusion, input/output and
hallucination guards, lexical Ragas scoring, and a timestamped HTML report.

Exit code is 0 when every demo query retrieves an answer and passes guards.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import math
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure the project root is importable when run as a script file.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog

from scripts.ingest_sample_data import ensure_sample_docs

logger = structlog.get_logger(__name__)

EMBED_DIM = 64

DEMO_QUERIES: list[tuple[str, str]] = [
    ("What chunk size and overlap does DeepResearch AI use?", "512"),
    ("What is the RRF k value used for hybrid retrieval fusion?", "60"),
    ("What Ragas metric targets must answers meet?", "0.85"),
]


class DemoEmbedder:
    """Deterministic trigram hash-bucket embeddings for key-free runs.

    Character-trigram hashes accumulate into fixed-size buckets, L2-normalized
    so cosine similarity reflects lexical overlap — including morphological
    variants (``chunk`` ~ ``chunks``). Stable across runs by design.
    """

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self._dim = dim

    def _embed_one(self, text: str) -> list[float]:
        buckets = [0.0] * self._dim
        words = text.lower().split()
        for word in words:
            grams = [word] if len(word) < 3 else [word[i : i + 3] for i in range(len(word) - 2)]
            for gram in grams:
                digest = hashlib.sha256(gram.encode()).digest()
                buckets[int.from_bytes(digest[:2]) % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in buckets)) or 1.0
        return [v / norm for v in buckets]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed raw strings."""
        return [self._embed_one(t) for t in texts]

    async def embed_documents(self, documents: list[object]) -> list[list[float]]:
        """Embed LangChain documents via their page content."""
        return [self._embed_one(str(getattr(d, "page_content", d))) for d in documents]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        return self._embed_one(query)


async def run_demo(report_dir: str = "data/eval_reports") -> bool:
    """Execute the full pipeline; return True when every query succeeds."""
    import chromadb

    from src.evaluation.datasets import EvalSample
    from src.evaluation.ragas_eval import RagasEvaluator
    from src.evaluation.reports import save_html_report, summarize_results
    from src.guardrails.input_guards import InputGuard
    from src.guardrails.output_guards import OutputGuard
    from src.guardrails.validators import HallucinationGuard
    from src.ingestion.pipeline import IngestionPipeline
    from src.retrieval.pipeline import RetrievalPipeline
    from src.vectorstore.chroma_store import ChromaStore

    ok = True
    # mkdtemp (not TemporaryDirectory): ChromaDB memory-maps its files on
    # Windows, so eager cleanup at exit raises PermissionError. Leftovers are
    # best-effort removed; the OS temp cleaner reaps the rest.
    tmp = tempfile.mkdtemp(prefix="dr-demo-")
    try:
        paths = ensure_sample_docs(Path(tmp) / "sample_docs")
        embedder = DemoEmbedder()

        # 1. Ingest fixtures (loader -> chunker -> dedup -> fake embed).
        ingestion = IngestionPipeline(embedder=embedder)  # type: ignore[arg-type]
        all_chunks = []
        for path in paths:
            all_chunks.extend(await ingestion.ingest(str(path)))
        print(f"ingest: {len(paths)} docs -> {len(all_chunks)} chunks")

        # 2. Store in an isolated ChromaDB (never touches production data).
        client = chromadb.PersistentClient(path=str(Path(tmp) / "chroma"))
        store = ChromaStore(collection_name="demo", client=client)
        stored = await store.upsert(all_chunks)
        print(f"store: upserted {stored} chunks")

        # 3. Retrieve + guard + score each demo query.
        texts = [c.text for c in all_chunks]
        metas = [c.metadata for c in all_chunks]
        retrieval = RetrievalPipeline(
            vector_store=store,
            embedder=embedder,  # type: ignore[arg-type]
            documents=texts,
            metadatas=metas,
            enable_sparse=True,
            enable_rerank=False,
            enable_multi_query=False,
        )
        input_guard = InputGuard()
        output_guard = OutputGuard()
        hallucination_guard = HallucinationGuard()
        evaluator = RagasEvaluator()
        samples: list[EvalSample] = []

        for question, must_contain in DEMO_QUERIES:
            in_guard = await input_guard.validate(question)
            result = await retrieval.retrieve(question, top_k=3)
            if not result.results:
                print(f"FAIL retrieve: {question!r} (no results)")
                ok = False
                continue
            top = result.results[0]
            answer = f"{top.text.strip()} [Source 1]"
            contexts = [r.text for r in result.results]
            sources = [{"title": top.metadata.source, "text": top.text}]

            out_guard = await output_guard.validate(answer, sources)
            hal_guard = await hallucination_guard.validate(answer, contexts)
            passed = bool(in_guard.passed and out_guard.passed and hal_guard.passed)
            # Recall assertion: the needed fact must be in the retrieved set,
            # not necessarily the single top chunk.
            retrieved_ok = any(must_contain in text for text in contexts)
            ok = ok and passed and retrieved_ok
            status = "PASS" if (passed and retrieved_ok) else "FAIL"
            print(f"{status} query: {question!r} (score={top.score:.3f})")

            samples.append(
                EvalSample(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=must_contain,
                )
            )

        # 4. Score and report (read-only over the demo run). Thresholds are the
        # production LLM-judge bars; the lexical fallback undershoots them by
        # design, so per-sample verdicts are informational here.
        try:
            import ragas  # noqa: F401

            eval_mode = "ragas"
        except ImportError:
            eval_mode = "lexical-fallback"
        eval_results = await evaluator.evaluate_batch(samples)
        summary = summarize_results(eval_results)
        report_path = save_html_report(eval_results, output_dir=report_dir)
        print(
            f"eval [{eval_mode}]: {summary.get('n_passed', 0)}/{summary.get('n', 0)} "
            f"pass production bars "
            f"(mean faithfulness={summary.get('mean_faithfulness', 'n/a')})"
        )
        print(f"report: {report_path}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ok


async def main(report_dir: str) -> int:
    """Entry point returning a process exit code."""
    return 0 if await run_demo(report_dir) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the key-free end-to-end demo.")
    parser.add_argument("--report-dir", default="data/eval_reports")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.report_dir)))
