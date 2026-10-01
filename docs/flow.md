# Data & Execution Flow

> **Audience:** novice / vibe coder. No prior RAG knowledge assumed.
> This is the companion to `docs/ARCHITECTURE.md` (which explains *what the
> system is*) — this doc explains *what happens, in what order, in which file*
> when data moves through the AI pipeline.

---

## 1. The one-picture flow

```
YOUR FILES                    KNOWLEDGE BASE                  YOUR QUESTION
   │                                │                               │
   ▼                                ▼                               ▼
┌──────────┐   chunks   ┌──────────────────┐   query   ┌─────────────────────┐
│ INGEST   │ ─────────▶ │ CHROMA (on disk) │ ◀──────── │ RETRIEVE            │
│ load →   │  +vectors  │ data/chroma_db/  │  vectors  │ expand → dense+BM25 │
│ chunk →  │            └──────────────────┘           │ → RRF fuse → rerank │
│ dedup →  │                                           └─────────┬───────────┘
│ embed    │                                                     │ top chunks
└──────────┘                                                     ▼
                                                       ┌─────────────────────┐
                                                       │ GENERATE answer     │
                                                       │ (LLM + citations)   │
                                                       └─────────┬───────────┘
                                                                 ▼
                                                       ┌─────────────────────┐
                                                       │ CRITIQUE loop       │
                                                       │ (max 3 tries to     │
                                                       │  fix bad answers)   │
                                                       └─────────┬───────────┘
                                                                 ▼
                                                       ┌─────────────────────┐
                                                       │ GUARDS check it     │
                                                       │ (safe? grounded?    │
                                                       │  cited?)            │
                                                       └─────────┬───────────┘
                                                                 ▼
                                                     answer + sources + score
```

**In one sentence:** your documents become searchable numbers (embeddings);
your question becomes numbers too; the system finds the closest numbers,
asks an LLM to answer *only from those*, critiques its own answer, checks it
for safety, and shows you the answer with sources.

---

## 2. Glossary (words you'll see everywhere)

| Word | Plain meaning | Example in this project |
|------|---------------|-------------------------|
| Chunk | A slice of a document (~512 tokens) | One row in ChromaDB |
| Embedding | Text turned into a list of numbers capturing meaning | `[0.12, -0.44, ...]` (64–1536 numbers) |
| Vector store | Database that finds *similar* numbers fast | ChromaDB folder `data/chroma_db/` |
| Dense retrieval | Search by meaning (embeddings) | `src/retrieval/dense.py` |
| Sparse/BM25 | Search by exact keywords | `src/retrieval/sparse.py` |
| RRF (k=60) | Voting system merging both ranked lists | `src/retrieval/hybrid.py` |
| Rerank | Second, smarter sort of the top candidates | `src/retrieval/reranker.py` |
| Guard | Safety checkpoint (pass/fail) wrapping agent I/O | `src/guardrails/` |
| Faithfulness | "Is every claim backed by retrieved text?" (target > 0.85) | Ragas metric |
| Loop guard | Hard cap so the AI can't retry forever (max 3) | `graph.py`, `review_loop.py` |

---

## 3. Stage-by-stage: what happens and where

### Stage A — INGEST (`src/ingestion/`, contract §4.1)
Turns raw files into `ProcessedChunk(id, text, embedding, metadata)` objects.

| Step | File:function | What it does |
|------|---------------|--------------|
| 1. Load | `loaders.py:DocumentLoader.load()` | Reads PDF / web / Markdown / CSV / text → LangChain Documents |
| 2. Chunk | `chunker.py:TextChunker.achunk_recursive()` | Splits into ~512-token pieces with 64-token overlap |
| 3. Dedup | `deduplicator.py` | Drops chunks with identical content hash |
| 4. Embed | `embedder.py:Embedder.embed_documents()` | Text → vectors via OpenAI (needs `OPENAI_API_KEY`) |
| 5. Wrap | `pipeline.py:IngestionPipeline.ingest()` | Runs 1–4; returns `list[ProcessedChunk]` |

### Stage B — STORE (`src/vectorstore/`, contract §4.2)
Saves chunks + vectors **on disk** so knowledge survives restarts.

| Step | File:function | What it does |
|------|---------------|--------------|
| 1. Connect | `chroma_store.py:_get_client()` | Opens `chromadb.PersistentClient(path=...)` — once per process |
| 2. Save | `chroma_store.py:ChromaStore.upsert()` | Writes ids + texts + vectors + metadata |
| 3. Access | `manager.py:get_store()` | Returns the shared store (ChromaDB, or Pinecone stub) |

### Stage C — RETRIEVE (`src/retrieval/`, contract §4.3)
Turns your question into the best supporting chunks. Entry:
`pipeline.py:RetrievalPipeline.retrieve(query, top_k=5)`.

| Step | File | What it does |
|------|------|--------------|
| 1. Expand | `multi_query.py` | Rewrites your question 2–3 ways for better recall |
| 2. Embed query | `embedder.py:embed_query()` | Question → vector |
| 3. Dense | `dense.py` | Top matches by *meaning* |
| 4. Sparse | `sparse.py` | Top matches by *keywords* (BM25) |
| 5. Fuse (RRF) | `hybrid.py` | Merges both lists: `score = Σ 1/(60 + rank)` |
| 6. Rerank | `reranker.py` | Cross-encoder re-sorts the fused shortlist |
| Output | `pipeline.py:RetrievalResult` | `query, expanded_queries, results, retrieval_metadata` |

### Stage D — GENERATE (`src/agents/generation.py`)
`GenerationPipeline.generate_answer(query, retrieval_result)` prompts the LLM
with the retrieved chunks and returns `{answer, sources[]}` with `[Source N]`
citations baked in.

### Stage E — CRITIQUE LOOP (`src/agents/`, max 3 iterations)
The agent checks its own answer and retries if it fails:

```
research → critique/review → revise → critique/review → … → deliver
   ▲                          │ fail + grounding gap → back to research
   └──────────────────────────┘ fail + bad wording → revise
```

| Piece | File | Role |
|-------|------|------|
| LangGraph machine | `graph.py` | `StateGraph`, conditional edges, recursion-limit guard |
| Lightweight machine | `graph_skeleton.py` | Same flow, pure Python, no framework overhead |
| Shared mechanics | `review_loop.py:ReviewLoopBase` | research/revise/merge/route logic (single copy) |
| 4-dimension critic | `self_critique.py` | faithfulness, relevancy, accuracy, completeness |
| 3-dimension reviewer | `review_agent.py` | grounding, relevancy, completeness |
| External critic | `crew.py:CriticAgent` | CrewAI judge via `kickoff_async()` (never blocks) |
| State | `state.py:AgentState` | Pydantic model; `research_results` *accumulates* per loop |

### Stage F — GUARDS (`src/guardrails/`, contract §4.4)
Middleware at graph **intake** (input) and **deliver** (output). Each returns
`GuardResult(passed, violations, action_taken)`.

| Guard | File:method | Blocks what |
|-------|-------------|-------------|
| Input | `input_guards.py:validate(query)` | Prompt injection, empty/over-long queries. Topic match is **advisory only** (logged, never blocks) |
| Output | `output_guards.py:validate(answer, sources)` | Missing citations, bad format, unsafe content |
| Hallucination | `validators.py:validate(answer, contexts)` | Claims not grounded in context (0–1 score) |

> **Design rule:** the gate checks *safety*, the pipeline checks *relevance*.
> A keyword gate can't tell paraphrase from off-topic, so it never rejects for
> topic. Instead the retrieval **coverage floor** (`DEFAULT_MIN_COVERAGE`,
> top-1 dense similarity) trips when the corpus doesn't cover the query — the
> agent then delivers an honest templated answer with **no LLM call**.

### Stage G — EVALUATE (`src/evaluation/`, contract §4.5)
Scores answers without ever touching production data.

| Step | File | What it does |
|------|------|--------------|
| Samples | `datasets.py:GoldenDatasetManager` | Golden Q&A triples (+ synthetic augmentation) |
| Score | `ragas_eval.py:RagasEvaluator.evaluate_batch()` | Real `ragas` lib if installed, else lexical fallback |
| Score (2nd opinion) | `deepeval_eval.py` | Independent hallucination metric |
| Report | `reports.py:save_html_report()` | Timestamped HTML → `data/eval_reports/` |

---

## 4. Execution traces (follow your code path)

### Trace 1 — Key-free demo: `uv run python scripts/demo.py`
Best first run: needs **no API keys**.
```
demo.py → ingest_sample_data.ensure_sample_docs()   # writes 3 fixtures
  → IngestionPipeline(embedder=DemoEmbedder)        # hash embeddings, no keys
  → ChromaStore(client=<temp dir>)                  # isolated, production untouched
  → RetrievalPipeline(...).retrieve(q)              # dense+BM25+RRF
  → InputGuard / OutputGuard / HallucinationGuard   # all must pass
  → RagasEvaluator.evaluate_batch()                 # lexical-fallback mode
  → save_html_report() → data/eval_reports/*.html   # exit 0 = all queries PASS
```

### Trace 2 — API question: `POST /api/v1/query {"query": "..."}`
```
api/main.py (app) → routes/query.py::query_knowledge_base()
  → 400 if query blank
  → _get_retrieval()      # process-wide singleton (built once, reused)
  → retrieve(query)       # Stage C
  → 404 if no results
  → _get_generator().generate_answer()  # Stage D
  → QueryResponse(answer, sources)
```

### Trace 3 — UI: `uv run streamlit run src/ui/app.py`
```
app.py (sidebar nav) → pages/research_chat.py  → POSTs to /api/v1/query
                     → pages/knowledge_base.py → uploads via /api/v1/ingest
                     → pages/eval_dashboard.py → previews data/eval_reports/*.html
```

### Trace 4 — Golden eval: `uv run python scripts/run_evals.py`
```
datasets.py (load golden Q&A) → answer each via generation pipeline
  → RagasEvaluator + DeepEval → save_html_report() → console summary
```

---

## 5. Worked example (one question, end to end)

Question: *“What chunk size does DeepResearch AI use?”*

1. **Guard in**: `InputGuard` — no injection, on-topic, short → pass.
   (Topic is advisory; a paraphrased question is never rejected here.)
2. **Expand**: `["What chunk size…?", "How are documents split…?", …]`.
3. **Embed**: question → vector.
4. **Dense** finds chunks about *splitting*; **BM25** finds chunks containing
   *“chunk”*; **RRF** merges → chunk with *“512 tokens with a 64-token
   overlap”* wins; **reranker** confirms order.
5. **Generate**: LLM answers *“512 tokens with 64-token overlap [Source 1]”*.
6. **Critique**: faithfulness/relevancy pass → no revise needed → deliver.
7. **Guards out**: citation present, grounded → pass.
8. **You see**: answer + collapsible *Sources* + eval score on the dashboard.

---

## 6. File status tracker

| Area | Files | Status |
|------|-------|--------|
| `src/config/` | settings, constants | ✅ verified |
| `src/ingestion/` | loaders, chunker, embedder, deduplicator, pipeline | ✅ 150-test gate |
| `src/vectorstore/` | base, chroma_store, manager, pinecone stub | ✅ persistent round-trip works |
| `src/retrieval/` | dense, sparse, hybrid, reranker, multi_query, pipeline | ✅ RRF math tested |
| `src/agents/` | graph, graph_skeleton, review_loop, self_critique, review_agent, crew, generation, llm_provider, memory, prompts, state | ✅ loop guards tested |
| `src/guardrails/` | models, input/output guards, validators | ✅ 28 pass+fail tests |
| `src/evaluation/` | datasets, ragas, deepeval, reports (+synthetic) | ✅ fallback tested; live judge needs keys |
| `src/finetuning/` | curator, formatter, trainer, evaluator, exporter | ⏸️ deferred (ADR-004) |
| `src/api/` | main, routes, middleware | ✅ singleton pipelines |
| `src/ui/` | app, chat, research_chat, knowledge_base, eval_dashboard | ✅ imports verified |
| `scripts/` | demo, ingest_sample_data, run_evals, generate_synthetic | ✅ demo exits 0 |

Gate (2026-09-22): **151/151 tests · ruff check+format clean · mypy strict clean**.
Python **3.13** pinned (`.python-version`); demo needs no keys, real ingestion needs `OPENAI_API_KEY`.
