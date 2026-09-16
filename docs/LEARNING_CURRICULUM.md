# Learning Curriculum — AI Engineer Path via DeepResearch AI

> **Goal:** Build deep understanding of RAG systems, LLM orchestration, and AI engineering
> through hands-on work on this project.

---

## How to Use This Curriculum

1. **Start with Level 1** — master fundamentals before moving on
2. **Ask the mentor agent** questions about anything you don't understand
3. **Read the code** alongside the explanations — theory without practice is hollow
4. **Take notes** — write down what you learn in your own words
5. **Revisit concepts** — understanding deepens with repetition

---

## Level 1: RAG Fundamentals (Week 1-2)

### 1.1 What is RAG?
- **Concept:** Retrieval-Augmented Generation — giving an LLM access to external knowledge
- **Why it matters:** LLMs have knowledge cutoffs; RAG lets them answer about YOUR data
- **In our project:** `src/ingestion/` loads documents, `src/retrieval/` finds relevant chunks, `src/agents/generation.py` generates answers
- **Ask the mentor:** "How does RAG differ from fine-tuning?"

### 1.2 How Embeddings Work
- **Concept:** Converting text into numerical vectors that capture meaning
- **Why it matters:** Computers can't compare text directly; vectors let us measure similarity
- **In our project:** `src/ingestion/embedder.py` — OpenAI and HuggingFace models
- **Key insight:** Similar meanings → nearby vectors in high-dimensional space
- **Ask the mentor:** "What happens when two different texts have similar embeddings?"

### 1.3 Vector Stores
- **Concept:** Specialized databases for storing and searching embeddings
- **Why it matters:** Finding the most similar documents to a query
- **In our project:** `src/vectorstore/chroma_store.py` (ChromaDB for dev), `src/vectorstore/pinecone_store.py` (Pinecone for prod)
- **Key insight:** Vector stores use HNSW indexing for fast approximate nearest neighbor search
- **Ask the mentor:** "Why do we need a special database? Can't we just use PostgreSQL?"

### 1.4 Chunking
- **Concept:** Splitting long documents into smaller pieces for retrieval
- **Why it matters:** LLMs have context windows; smaller chunks = more precise retrieval
- **In our project:** `src/ingestion/chunker.py` — recursive and semantic strategies
- **Key insight:** Chunk size is a tradeoff: too small loses context, too large loses precision
- **Ask the mentor:** "What's the difference between recursive and semantic chunking?"

### 1.5 Retrieval Strategies
- **Dense retrieval:** Semantic search using embeddings (finds meaning)
- **Sparse retrieval (BM25):** Keyword search (finds exact terms)
- **Hybrid:** Combines both using Reciprocal Rank Fusion (RRF)
- **Reranking:** Cross-encoder re-scores results for precision
- **In our project:** `src/retrieval/dense.py`, `src/retrieval/sparse.py`, `src/retrieval/hybrid.py`, `src/retrieval/reranker.py`
- **Ask the mentor:** "Why not just use dense retrieval? What does BM25 add?"

---

## Level 2: The Stack (Week 3-4)

### 2.1 LangChain
- **What it is:** A toolkit for building LLM applications (chains, retrievers, loaders)
- **In our project:** Document loaders, text splitters, retriever interfaces
- **Key insight:** LangChain is a toolkit, not a framework — use specific features, don't build everything on top of it
- **Ask the mentor:** "What's the difference between a LangChain chain and a simple function?"

### 2.2 LangGraph
- **What it is:** A state machine framework for multi-step agent workflows
- **In our project:** `src/agents/graph_skeleton.py` — the review agent state machine
- **Key concept:** Nodes are functions, edges are conditional, state is the single source of truth
- **Key insight:** LangGraph handles loops, branches, and persistence that raw Python can't easily do
- **Ask the mentor:** "Why use LangGraph instead of just calling functions in order?"

### 2.3 CrewAI
- **What it is:** Role-based agent definitions with tools and delegation
- **In our project:** Deferred — we're testing if single-agent is sufficient first
- **Key concept:** Each agent has a role, goal, backstory, and toolset
- **Ask the mentor:** "What problem does CrewAI solve that LangGraph doesn't?"

### 2.4 Guardrails AI
- **What it is:** Input/output validation middleware for LLM calls
- **In our project:** Deferred — custom validators first, framework if complexity demands
- **Key concept:** Validators check LLM output, fixers attempt correction
- **Ask the mentor:** "What guardrails do we actually need for a research system?"

### 2.5 Evaluation (Ragas)
- **What it is:** RAG-specific quality metrics (faithfulness, relevancy, precision, recall)
- **In our project:** `src/evaluation/ragas_eval.py` — just built!
- **Key metrics:**
  - Faithfulness: Is the answer grounded in retrieved context?
  - Answer relevancy: Does the answer address the query?
  - Context precision: Are the retrieved chunks relevant?
  - Context recall: Did we retrieve all necessary information?
- **Ask the mentor:** "How do you interpret Ragas scores? What's a good baseline?"

---

## Level 3: Architecture & Design (Week 5-6)

### 3.1 Why This Architecture?
- **Read:** `docs/decisions.md` — all architectural decisions with reasoning
- **Key decision:** "Prove Before You Build" — evaluation first, multi-agent second
- **Ask the mentor:** "What was the council's main concern about the original architecture?"

### 3.2 Interface Contracts
- **Read:** `docs/CONTRACTS.md` — how modules communicate
- **Key concept:** Contracts are stable API surfaces; implementations can change freely
- **Ask the mentor:** "Why define contracts before implementation?"

### 3.3 The Council Deliberation
- **Read:** `docs/meetings/council-2026-09-13-architecture.md`
- **Key insights:** 6 historical figures debated the architecture — each brought a different lens
- **Ask the mentor:** "What was Torvalds' main criticism? Do you agree?"

---

## Level 4: Advanced Topics (Week 7+)

### 4.1 Self-Critique Patterns
- **Concept:** An agent reviews and improves its own output
- **In our project:** `src/agents/review_agent.py` — generate, review, revise loop
- **Ask the mentor:** "How does the review agent know when an answer is good enough?"

### 4.2 Multi-Agent Orchestration
- **Concept:** Multiple specialized agents collaborating on a task
- **In our project:** Planned for Phase 2 (if evaluation proves it's needed)
- **Ask the mentor:** "When is multi-agent better than single-agent?"

### 4.3 Fine-Tuning
- **Concept:** Training a smaller model on domain-specific data
- **In our project:** Phase 4 — QLoRA training on high-scoring Q&A pairs
- **Ask the mentor:** "When should you fine-tune vs. use RAG?"

### 4.4 Evaluation-Driven Development
- **Concept:** Using metrics to guide architectural decisions
- **In our project:** ADR-003 — build evaluation before adding complexity
- **Ask the mentor:** "How do you know when your system is good enough?"

---

## Suggested Learning Flow

```
Week 1:  RAG fundamentals (1.1-1.5) + read src/ingestion/
Week 2:  Vector stores + retrieval (1.3-1.5) + read src/retrieval/
Week 3:  LangChain basics (2.1) + read src/agents/generation.py
Week 4:  LangGraph + state machines (2.2) + read src/agents/graph_skeleton.py
Week 5:  Architecture decisions (3.1-3.3) + read docs/decisions.md
Week 6:  Evaluation (2.5) + read src/evaluation/
Week 7:  Self-critique (4.1) + read src/agents/review_agent.py
Week 8+: Advanced topics based on interest
```

---

## Questions to Ask the Mentor

### Beginner
- "How does our RAG pipeline work end-to-end?"
- "What is embedding and why do we need it?"
- "How does ChromaDB store and search vectors?"
- "What is the difference between dense and sparse retrieval?"

### Intermediate
- "How does RRF fusion combine dense and sparse results?"
- "Why did we choose ChromaDB over Pinecone for dev?"
- "How does the self-critique agent improve answers?"
- "What are Ragas metrics and how do we interpret them?"

### Advanced
- "Why did the council defer CrewAI?"
- "When is multi-agent worth the complexity?"
- "How do we handle LLM failures in the graph?"
- "What's the path from ChromaDB to Pinecone production?"

---

> **Remember:** The mentor agent is your learning companion. Ask anything, no question is too basic.
> The goal is understanding, not speed.
