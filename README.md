# DeepResearch AI

**Multi-agent research platform with a persistent knowledge base.**

Ingest documents → build a ChromaDB-backed knowledge base → ask questions through a
LangGraph-orchestrated retrieve → generate → critique → revise loop → get cited,
guardrailed, evaluated answers via FastAPI or Streamlit.

## Architecture

```
User ──▶ Streamlit UI ──▶ FastAPI ──▶ Retrieval Pipeline ──▶ ChromaDB (persistent)
         │                              │  dense + BM25 + RRF(k=60) + rerank
         │                              ▼
         │                         Generation (LLM provider abstraction)
         │                              │
         │                    ┌─────────▼──────────┐
         │                    │ LangGraph loop     │
         │                    │ intake→research→   │
         │                    │ critique→revise→   │
         │                    │ deliver (max 3)    │
         │                    └─────────┬──────────┘
         │              ┌───────────────┼───────────────┐
         │         InputGuard     OutputGuard   HallucinationGuard
         │              └───────────────┼───────────────┘
         │                              ▼
         └◀── cited answer ── Ragas/DeepEval scored ── HTML report
```

Module ownership, interface contracts, and ADRs: `AGENTS.md`, `docs/CONTRACTS.md`,
`docs/decisions.md`. Data flow: `docs/ARCHITECTURE.md`.

## Quick Start

```bash
# Python 3.13 is pinned (.python-version); uv fetches it automatically
uv sync --extra dev
cp .env.example .env   # add OPENAI_API_KEY / ANTHROPIC_API_KEY

# 1. Key-free end-to-end smoke test (ingest → retrieve → guards → eval report)
uv run python scripts/demo.py

# 2. Ingest your own documents (needs OPENAI_API_KEY)
uv run python scripts/ingest_sample_data.py

# 3. Serve the API + UI
uv run python -m src.api.main          # FastAPI on :8000
uv run streamlit run src/ui/app.py     # UI on :8501
```

## Verify

```bash
uv run pytest              # 150 tests, all green
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
uv run mypy src/           # strict, 61 files clean
uv run python scripts/run_evals.py     # golden-dataset eval + HTML report
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Liveness + version |
| `/api/v1/ingest` | POST | Ingest a document (`multipart/form-data`) |
| `/api/v1/query` | POST | `{"query": "..."}` → `{answer, sources[]}` (404 when nothing relevant) |

## Configuration

All settings come from environment (`.env`) via Pydantic Settings
(`src/config/settings.py`): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`PINECONE_API_KEY` (optional), `EMBEDDING_MODEL`, `LLM_MODEL`, `CHUNK_SIZE`
(512), `CHUNK_OVERLAP` (64), `CHROMA_PERSIST_DIR` (`./data/chroma_db`),
`TOP_K` (5). No secrets in code — ever.

## Evaluation

Golden Q&A lives in `data/golden_datasets/`. `RagasEvaluator` prefers the real
`ragas` library and degrades to deterministic lexical scorers offline; verdicts
require faithfulness > 0.85 and answer relevancy > 0.90. Every run saves a
timestamped HTML report to `data/eval_reports/`. Evaluation reads production
data but never modifies it.

## Production Notes

- ChromaDB uses `PersistentClient` — knowledge survives restarts.
- API pipelines are process-wide singletons (no per-request rebuild).
- All I/O is `asyncio`; CrewAI review runs via `kickoff_async()`.
- Guards are middleware, not inline — agents work identically with them off.
- Status: Phase 1–3 built; fine-tuning (Phase 4) deferred by design (ADR-004).

## License

MIT
