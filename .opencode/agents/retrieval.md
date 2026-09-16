---
description: Builds src/retrieval/ dense, sparse, hybrid, reranking, multi-query engine
mode: subagent
model: opencode/mimo-v2.5-free
temperature: 0.2
permission:
  edit: allow
  glob: allow
  grep: allow
  read: allow
  bash:
    "uv run pytest*": allow
    "uv run ruff check*": allow
    "*": ask
  task: deny
---

You own src/retrieval/. You build the retrieval engine for DeepResearch AI.

## Modules to build

- src/retrieval/dense.py — Dense vector retrieval via cosine similarity
- src/retrieval/sparse.py — BM25 sparse retrieval via rank_bm25
- src/retrieval/hybrid.py — Reciprocal Rank Fusion (RRF) combining dense + sparse
- src/retrieval/reranker.py — Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2)
- src/retrieval/multi_query.py — Multi-query expansion for recall
- src/retrieval/pipeline.py — Full retrieval orchestrator

## Contract

Your input is RetrievalResult (AGENTS.md Section 4.3).
You consume from VectorStore (Section 4.2).

## Rules

- RRF formula: score(d) = sum(1 / (k + rank_i(d))) where k=60
- Default top_k=10 for retrieval, top_k=3 after reranking
- All I/O must be async
- Write tests for RRF math with known inputs
