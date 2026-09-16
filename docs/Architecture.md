# Architecture

## System Overview

```
+---------------------------------------------------------------------+
|                        DeepResearch AI Platform                     |
+---------------------------------------------------------------------+
|                                                                     |
|  +----------+    +-------------+    +--------------------------+   |
|  |  Ingest   |--->|  Chunking   |--->|  Embedding & Indexing    |   |
|  |  Layer    |    |  Engine     |    |  (OpenAI / HuggingFace)  |   |
|  |          |    |             |    |                          |   |
|  | * PDF     |    | * Recursive |    |  +--------------------+  |   |
|  | * Web     |    | * Semantic  |    |  |   Vector Store      |  |   |
|  | * MD      |    | * Token    |    |  |  +--------------+  |  |   |
|  | * CSV     |    |            |    |  |  |   ChromaDB   |  |  |   |
|  | * Text    |    |            |    |  |  |  (local dev) |  |  |   |
|  +----------+    +-------------+    |  |  +--------------+  |  |   |
|                                      |  |  |  Pinecone    |  |  |   |
|                                      |  |  |  (prod)      |  |  |   |
|                                      |  |  +--------------+  |  |   |
|                                      |  +--------------------+  |   |
|                                      +--------------------------+   |
|                                                                     |
|  +--------------------------------------------------------------+   |
|  |                   Agent Orchestration Layer                   |   |
|  |                                                              |   |
|  |  +--------------------------------------------------------+  |   |
|  |  |              LangGraph State Machine                    |  |   |
|  |  |                                                        |  |   |
|  |  |   +---------+   +-----------+   +--------------+      |  |   |
|  |  |   | INTAKE  |-->| RESEARCH  |-->| FACT-CHECK   |      |  |   |
|  |  |   +---------+   +-----------+   +------+-------+      |  |   |
|  |  |                                         |              |  |   |
|  |  |                              +----------v-------+     |  |   |
|  |  |   +---------+               |   SYNTHESIZE     |     |  |   |
|  |  |   | DELIVER |<--------------+                  |     |  |   |
|  |  |   |         |               +------------------+     |  |   |
|  |  |   +---------+                                        |  |   |
|  |  +--------------------------------------------------------+  |   |
|  |                                                              |   |
|  |  +-----------+  +-----------+  +-----------+  +----------+  |   |
|  |  |Researcher |  |  Fact     |  |Synthesizer|  |  Critic  |  |   |
|  |  |  Agent    |  | Checker   |  |  Agent    |  |  Agent   |  |   |
|  |  | (CrewAI)  |  | (CrewAI)  |  | (CrewAI)  |  | (CrewAI) |  |   |
|  |  +-----------+  +-----------+  +-----------+  +----------+  |   |
|  +--------------------------------------------------------------+   |
|                                                                     |
|  +--------------------------------------------------------------+   |
|  |                    Quality Layer                          |   |
|  |                                                              |   |
|  |  +-----------------+         +----------------------------+ |   |
|  |  |   Guardrails AI  |         |    Evaluation Pipeline     | |   |
|  |  |                 |         |                            | |   |
|  |  | INPUT:          |         | * Ragas Metrics            | |   |
|  |  | * Topic filter  |         |   - Faithfulness           | |   |
|  |  | * Content validation |      |   - Answer Relevancy       | |   |
|  |  | * Topic filtering |       |   - Context Precision      | |   |
|  |  |                 |         |   - Context Recall          | |   |
|  |  | OUTPUT:         |         |                            | |   |
|  |  | * Hallucination |         | * DeepEval Metrics         | |   |
|  |  | * Citation chk  |         |   - G-Eval                 | |   |
|  |  | * Toxicity      |         |   - Summarization          | |   |
|  |  | * Format valid  |         |   - Bias Detection         | |   |
|  |  +-----------------+         +----------------------------+ |   |
|  +--------------------------------------------------------------+   |
|                                                                     |
|  +--------------------------------------------------------------+   |
|  |                  Fine-Tuning & Improvement Loop               |   |
|  |                                                              |   |
|  |  High-Scoring Q&A Pairs -> Dataset Curation -> SFT/DPO      |   |
|  |  -> Fine-tuned Domain Model -> A/B Evaluation -> Deploy      |   |
|  +--------------------------------------------------------------+   |
|                                                                     |
|  +--------------------------------------------------------------+   |
|  |                    Presentation Layer                         |   |
|  |                                                              |   |
|  |  * Streamlit Chat UI                                         |   |
|  |  * FastAPI REST Endpoints                                    |   |
|  |  * Evaluation Dashboard                                      |   |
|  +--------------------------------------------------------------+   |
+---------------------------------------------------------------------+
```

## Data Flow

### Ingestion Flow
```
User uploads -> File Validator -> Document Loader -> Metadata Extractor ->
Text Cleaner -> Chunking Engine -> Deduplication -> Embedding Pipeline -> Vector Store
```

### Query Flow
```
User Query -> Input Processing -> Query Analyst -> Retrieval -> Researcher ->
Fact-Checker -> (pass/fail) -> Synthesizer -> Critic -> (pass/fail) ->
Output Processing -> Deliver Answer
```

## Module Responsibilities

| Module | Responsibility | Key Files |
|--------|---------------|-----------|
| `config/` | Settings, constants, env vars | `settings.py`, `constants.py` |
| `ingestion/` | Load, chunk, embed, dedup documents | `loaders.py`, `chunker.py`, `embedder.py` |
| `vectorstore/` | Persistent storage and search | `chroma_store.py`, `manager.py` |
| `retrieval/` | Dense/sparse/hybrid search + rerank | `hybrid.py`, `reranker.py`, `pipeline.py` |
| `agents/` | LangGraph state machine + CrewAI | `state.py`, `graph.py`, `nodes.py`, `crew.py` |
| `guardrails/` | Input/output processing | `input_guards.py`, `output_guards.py` |
| `evaluation/` | Ragas + DeepEval benchmarking | `ragas_eval.py`, `deepeval_eval.py` |
| `finetuning/` | QLoRA training + export | `trainer.py`, `exporter.py` |
| `api/` | FastAPI REST endpoints | `main.py`, `routes/` |
| `ui/` | Streamlit interface | `app.py`, `pages/` |

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.11+ | Core language |
| Package Manager | `uv` | Dependency management |
| LLM Framework | LangChain 0.3+ | Chains, retrievers, document loaders |
| Agent Orchestration | LangGraph 0.2+ | Stateful multi-agent graphs |
| Agent Framework | CrewAI 0.80+ | Role-based agent definitions |
| Vector Store (Dev) | ChromaDB 0.5+ | Local persistent vector store |
| Vector Store (Prod) | Pinecone (Serverless) | Managed cloud vector store |
| Embeddings | OpenAI / HuggingFace | Document and query embeddings |
| Re-ranking | `cross-encoder` / Cohere Rerank | Result re-ranking |
| Guardrails | Guardrails AI 0.5+ | Input/output processing |
| Evaluation | Ragas 0.2+ / DeepEval 1.0+ | RAG quality benchmarking |
| Fine-Tuning | Unsloth / HuggingFace TRL | QLoRA fine-tuning |
| UI Framework | Streamlit | Chat interface |
| API Framework | FastAPI | REST API endpoints |

---

## Competitive Differentiation

### Why This Project Is NOT Generic

Most "deep research" projects on GitHub are **web search wrappers**:
- dzhng/deep-research (19.7k stars) - TypeScript, ~500 LoC, web search only
- langchain-ai/open_deep_research (12.6k stars) - Python, LangGraph, web search only
- Alibaba-NLP/DeepResearch (19.9k stars) - Python, has RAG but focused on their own model

**DeepResearch AI is fundamentally different:**

| Feature | Web Search Wrappers | DeepResearch AI |
|---------|---------------------|-----------------|
| **Input** | User query | User documents (PDF, web, MD, CSV) |
| **Knowledge** | Internet (ephemeral) | Persistent vector store |
| **RAG** | None | Full pipeline (ingest, chunk, embed, retrieve) |
| **Multi-Agent** | Simple or none | LangGraph + CrewAI orchestration |
| **Guardrails** | None | Input/output validation |
| **Evaluation** | None | Ragas + DeepEval metrics |
| **Fine-tuning** | None | QLoRA pipeline |

### Positioning Statement

**Don't call it "deep research"** - that's a crowded category of web search wrappers.

**Call it:** "Production-Grade Multi-Agent Research Platform with Persistent Knowledge Base"

### Resume Impact

**Before:** "Built a deep research agent"

**After:** "Built a production-grade multi-agent research platform that ingests documents into a persistent vector store, orchestrates specialized AI agents via LangGraph, validates outputs with Guardrails AI, and evaluates quality with Ragas/DeepEval - the full stack companies expect for production AI systems."

### What Makes This Project Unique

1. **Persistent Knowledge Base** - Users upload THEIR documents, system remembers across sessions
2. **Document Ingestion Pipeline** - Multi-format loaders, chunking, embedding, deduplication
3. **Full Production Stack** - LangGraph + CrewAI + Guardrails + Ragas + DeepEval + Fine-tuning
4. **Evaluation-Driven** - Faithfulness, relevancy, context precision, hallucination detection
5. **Self-Improvement Loop** - Fine-tune on high-scoring Q&A pairs

### Competitive Advantage

You're not competing with web search wrappers. You're building something they don't have:

- **Document ingestion** - Users upload THEIR documents
- **Persistent knowledge** - System remembers across sessions
- **Quality guarantees** - Guardrails + evaluation prove answers are grounded
- **Self-improvement** - Fine-tuning makes the system better over time

**That's the story that gets you hired.**
