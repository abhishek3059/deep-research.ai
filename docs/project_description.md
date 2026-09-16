# 🧠 DeepResearch AI — Multi-Agent Research Platform

## Project Title
**DeepResearch AI** — *Autonomous Multi-Agent Research Assistant with Persistent Knowledge Base, Guardrailed Outputs, and Self-Improving Evaluation Loop*

---

## 📋 Table of Contents

- [Executive Summary](#executive-summary)
- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Core Architecture](#core-architecture)
- [Feature Breakdown by Phase](#feature-breakdown-by-phase)
- [Technology Stack](#technology-stack)
- [Data Flow & Pipeline Architecture](#data-flow--pipeline-architecture)
- [Agent Design Specification](#agent-design-specification)
- [Vector Store Strategy](#vector-store-strategy)
- [Guardrails Specification](#processing-specification)
- [Evaluation Framework](#evaluation-framework)
- [Fine-Tuning Pipeline](#fine-tuning-pipeline)
- [API & UI Specification](#api--ui-specification)
- [Directory Structure](#directory-structure)
- [Development Prompt](#-development-prompt)

---

## Executive Summary

**DeepResearch AI** is a production-grade, multi-agent research platform that enables users to ingest large document collections (PDFs, web pages, markdown, Notion exports, academic papers), build a persistent searchable knowledge base backed by a real vector store, and query that knowledge through a team of specialized AI agents that collaborate via stateful graph orchestration.

Unlike simple RAG chatbots, DeepResearch AI employs:
- **Multiple specialized agents** (Researcher, Fact-Checker, Synthesizer, Critic) orchestrated via LangGraph state machines
- **CrewAI role-based delegation** for complex multi-step research workflows
- **Input and output processing** to ensure consistency, data handling, and topic relevance
- **Automated evaluation pipelines** that continuously benchmark retrieval and generation quality
- **A self-improving fine-tuning loop** where high-scoring Q&A pairs are used to fine-tune a smaller domain-specific model

The system is designed to demonstrate mastery of the modern AI/LLM engineering stack in a single, cohesive, portfolio-worthy project.

---

## Problem Statement

### The Knowledge Worker's Pain

Knowledge workers — researchers, analysts, engineers, students — deal with an overwhelming volume of documents daily. Current tools force them to either:

1. **Manually search** through documents (slow, error-prone, no semantic understanding)
2. **Use basic RAG chatbots** that retrieve chunks and generate answers with no quality control, no verification, and no multi-step reasoning
3. **Trust LLMs blindly** with no processing against inaccurate outputs, no evaluation of answer quality, and no mechanism for continuous improvement

### The AI Engineer's Gap

Most AI portfolio projects stop at "I built a chatbot with RAG." This project addresses the gap between toy demos and production systems by incorporating:
- Real persistent vector stores (not in-memory arrays)
- Multi-agent collaboration (not a single chain)
- Output quality processing (not blind trust)
- Quantitative evaluation (not vibes-based testing)
- Model improvement via fine-tuning (not static prompts)

---

## Solution Overview

DeepResearch AI is a **4-phase project** that builds incrementally from a solid RAG foundation to a full-stack AI research platform:

| Phase | Name | Core Deliverable |
|-------|------|------------------|
| **Phase 1** | Foundation RAG Pipeline | Document ingestion → chunking → embedding → vector store → retrieval → generation |
| **Phase 2** | Multi-Agent Orchestration | LangGraph state machine + CrewAI agent team with specialized roles |
| **Phase 3** | Processing & Evaluation | Input/output processing + automated quality benchmarking with Ragas/DeepEval |
| **Phase 4** | Fine-Tuning & Self-Improvement | Curate golden dataset from high-scoring pairs → fine-tune smaller model → deploy |

Each phase produces a **working, deployable system** — the project is never in an incomplete state.

---

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DeepResearch AI Platform                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────────────┐   │
│  │  Ingest   │───▶│  Chunking   │───▶│  Embedding & Indexing    │   │
│  │  Layer    │    │  Engine     │    │  (OpenAI / HuggingFace)  │   │
│  │          │    │             │    │                          │   │
│  │ • PDF     │    │ • Recursive │    │  ┌────────────────────┐  │   │
│  │ • Web     │    │ • Semantic  │    │  │   Vector Store      │  │   │
│  │ • MD      │    │ • Token    │    │  │  ┌──────────────┐  │  │   │
│  │ • Notion  │    │ • Agentic  │    │  │  │   ChromaDB   │  │  │   │
│  │ • CSV     │    │            │    │  │  │  (local dev) │  │  │   │
│  │ • Arxiv   │    │            │    │  │  ├──────────────┤  │  │   │
│  └──────────┘    └─────────────┘    │  │  │  Pinecone    │  │  │   │
│                                      │  │  │  (prod)      │  │  │   │
│                                      │  │  └──────────────┘  │  │   │
│                                      │  └────────────────────┘  │   │
│                                      └──────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   Agent Orchestration Layer                   │   │
│  │                                                              │   │
│  │  ┌────────────────────────────────────────────────────────┐  │   │
│  │  │              LangGraph State Machine                    │  │   │
│  │  │                                                        │  │   │
│  │  │   ┌─────────┐   ┌───────────┐   ┌──────────────┐     │  │   │
│  │  │   │ INTAKE  │──▶│ RESEARCH  │──▶│ FACT-CHECK   │     │  │   │
│  │  │   │         │   │           │   │              │     │  │   │
│  │  │   └─────────┘   └───────────┘   └──────┬───────┘     │  │   │
│  │  │                                         │              │  │   │
│  │  │                              ┌──────────▼───────┐     │  │   │
│  │  │   ┌─────────┐               │   SYNTHESIZE     │     │  │   │
│  │  │   │ DELIVER │◀──────────────│                  │     │  │   │
│  │  │   │         │               └──────────────────┘     │  │   │
│  │  │   └─────────┘                                        │  │   │
│  │  └────────────────────────────────────────────────────────┘  │   │
│  │                                                              │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────┐ │   │
│  │  │Researcher │  │  Fact     │  │Synthesizer│  │  Critic  │ │   │
│  │  │  Agent    │  │ Checker   │  │  Agent    │  │  Agent   │ │   │
│  │  │ (CrewAI)  │  │ (CrewAI)  │  │ (CrewAI)  │  │ (CrewAI) │ │   │
│  │  └───────────┘  └───────────┘  └───────────┘  └──────────┘ │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Quality Layer                          │   │
│  │                                                              │   │
│  │  ┌─────────────────┐         ┌────────────────────────────┐ │   │
│  │  │   Guardrails AI  │         │    Evaluation Pipeline     │ │   │
│  │  │                 │         │                            │ │   │
│  │  │ INPUT:          │         │ • Ragas Metrics            │ │   │
│  │  │ • Topic filter  │         │   - Faithfulness           │ │   │
│  │  │ • Input valid.  │         │   - Answer Relevancy       │ │   │
│  │  │ • Adversarial   │         │   - Context Precision      │ │   │
│  │  │   input detect  │         │   - Context Recall          │ │   │
│  │  │ OUTPUT:         │         │                            │ │   │
│  │  │ • Accuracy chk  │         │ • DeepEval Metrics         │ │   │
│  │  │ • Citation chk  │         │   - G-Eval                 │ │   │
│  │  │ • Content qual. │         │   - Summarization          │ │   │
│  │  │ • Format valid  │         │   - Bias Detection         │ │   │
│  │  └─────────────────┘         └────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  Fine-Tuning & Improvement Loop               │   │
│  │                                                              │   │
│  │  High-Scoring Q&A Pairs → Dataset Curation → SFT/DPO        │   │
│  │  → Fine-tuned Domain Model → A/B Evaluation → Deploy         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Presentation Layer                         │   │
│  │                                                              │   │
│  │  • Streamlit / Chainlit Chat UI                              │   │
│  │  • FastAPI REST Endpoints                                    │   │
│  │  • LangSmith / Arize Phoenix Observability Dashboard         │   │
│  │  • Evaluation Leaderboard & Metrics Dashboard                │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Feature Breakdown by Phase

### Phase 1: Foundation RAG Pipeline

> **Goal:** Build a robust, production-quality RAG pipeline with a real persistent vector store.

#### 1.1 Document Ingestion Layer
- **Multi-format loaders** via LangChain:
  - `PyPDFLoader` / `UnstructuredPDFLoader` for PDFs
  - `WebBaseLoader` + `AsyncHtmlLoader` for web pages
  - `NotionDirectoryLoader` for Notion exports
  - `CSVLoader` for structured data
  - `ArxivLoader` for academic papers
  - `UnstructuredMarkdownLoader` for markdown
- **Metadata extraction:** file name, source URL, page number, section headers, ingestion timestamp
- **Deduplication:** content-hash-based dedup to prevent re-indexing identical chunks

#### 1.2 Chunking Engine
- **Recursive Character Text Splitter** as baseline
- **Semantic Chunking** via `SemanticChunker` (splits by embedding similarity)
- **Configurable parameters:**
  - `chunk_size`: 512 tokens (default)
  - `chunk_overlap`: 64 tokens (default)
  - `separators`: hierarchical (`\n\n`, `\n`, `. `, ` `)
- **Parent-Child Document Retrieval:** store both large parent chunks and smaller child chunks; retrieve children, return parents for better context

#### 1.3 Embedding Pipeline
- **Primary:** OpenAI `text-embedding-3-small` (1536d) or `text-embedding-3-large` (3072d)
- **Fallback/Local:** HuggingFace `BAAI/bge-small-en-v1.5` via `sentence-transformers`
- **Batch processing** with configurable concurrency and rate limiting
- **Embedding cache** to avoid re-computing unchanged chunks

#### 1.4 Vector Store (Persistent)
- **Development:** ChromaDB (local, persistent mode with SQLite backend)
  - Collections with metadata filtering
  - HNSW index for approximate nearest neighbor
- **Production:** Pinecone (managed, serverless)
  - Namespaces for multi-tenant isolation
  - Metadata filtering with complex queries
  - Hybrid search (dense + sparse via SPLADE)
- **Abstraction layer:** `VectorStoreManager` class that switches between backends via config

#### 1.5 Retrieval Engine
- **Dense retrieval:** Cosine similarity via vector store
- **Sparse retrieval:** BM25 via `rank_bm25`
- **Hybrid fusion:** Reciprocal Rank Fusion (RRF) combining dense + sparse results
- **Re-ranking:** Cross-encoder re-ranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for final ranking
- **Configurable `top_k`** with sensible defaults (k=5 for retrieval, k=3 after re-ranking)
- **Multi-query retrieval:** Generate multiple phrasings of the user query to improve recall
- **Self-query retrieval:** LLM extracts metadata filters from natural language queries

#### 1.6 Generation Pipeline
- **LLM providers:** OpenAI GPT-4o / GPT-4o-mini, Anthropic Claude, local Ollama (Llama 3, Mistral)
- **Streaming support** via SSE for real-time token delivery
- **System prompt engineering:** Grounded-answer prompts with explicit instructions to cite sources and refuse when context is insufficient
- **Conversation memory:** `ConversationBufferWindowMemory` with configurable window size

---

### Phase 2: Multi-Agent Orchestration

> **Goal:** Replace the single-chain RAG with a team of specialized agents orchestrated via a stateful graph.

#### 2.1 LangGraph State Machine
Define a directed graph with the following nodes:

```python
class ResearchState(TypedDict):
    query: str                          # Original user query
    refined_query: str                  # Query after decomposition/refinement
    sub_queries: list[str]              # Decomposed sub-questions
    retrieved_contexts: list[Document]  # Retrieved chunks from vector store
    research_notes: str                 # Researcher's raw findings
    fact_check_report: str              # Fact-checker's verification report
    fact_check_passed: bool             # Whether facts were verified
    synthesis: str                      # Synthesized final answer
    critique: str                       # Critic's feedback
    critique_passed: bool               # Whether critique was satisfactory
    citations: list[Citation]           # Extracted source citations
    final_answer: str                   # Delivered answer
    iteration_count: int                # Loop counter for critique cycles
    max_iterations: int                 # Max allowed critique loops (default: 3)
```

**Graph Topology:**

```
INTAKE → DECOMPOSE → RESEARCH → FACT_CHECK
                                      │
                          ┌───────────┤
                          │    PASS   │  FAIL
                          ▼           ▼
                     SYNTHESIZE   RESEARCH (retry with refined query)
                          │
                          ▼
                       CRITIQUE
                          │
                  ┌───────┤
                  │ PASS  │ FAIL (iteration < max)
                  ▼       ▼
              DELIVER   SYNTHESIZE (revise with feedback)
```

- **Conditional edges** for branching logic (fact-check pass/fail, critique pass/fail)
- **Loop guards** to prevent infinite cycles (max 3 iterations)
- **State persistence** via LangGraph checkpointing for resumable sessions

#### 2.2 CrewAI Agent Definitions

**Agent 1: Query Analyst**
- **Role:** Understand, decompose, and refine the user's research question
- **Capabilities:** Query decomposition into sub-questions, intent classification, metadata filter extraction
- **Tools:** None (pure reasoning)

**Agent 2: Researcher**
- **Role:** Deep-dive into the vector store to find relevant information
- **Capabilities:** Multi-query search, iterative retrieval, chunk summarization
- **Tools:** `VectorStoreSearchTool`, `WebSearchTool` (optional, for supplementary info)

**Agent 3: Fact-Checker**
- **Role:** Verify claims made by the Researcher against source documents
- **Capabilities:** Claim extraction, source-claim alignment scoring, contradiction detection
- **Tools:** `VectorStoreSearchTool` (independent search to cross-verify)

**Agent 4: Synthesizer**
- **Role:** Combine verified research into a coherent, well-structured answer
- **Capabilities:** Multi-source synthesis, citation formatting, structured output generation
- **Tools:** None (pure generation)

**Agent 5: Critic**
- **Role:** Evaluate the synthesized answer for quality, completeness, and accuracy
- **Capabilities:** Gap analysis, coherence scoring, accuracy verification
- **Tools:** None (pure evaluation)

#### 2.3 CrewAI Crew Configuration

```python
crew = Crew(
    agents=[query_analyst, researcher, fact_checker, synthesizer, critic],
    tasks=[analyze_task, research_task, verify_task, synthesize_task, critique_task],
    process=Process.sequential,  # or Process.hierarchical with a manager
    memory=True,                 # Enable shared crew memory
    cache=True,                  # Cache repeated tool calls
    verbose=True,                # Debug logging
    max_rpm=30,                  # Rate limiting
)
```

---

### Phase 3: Guardrails & Evaluation

> **Goal:** Add input/output quality gates and quantitative evaluation benchmarks.

#### 3.1 Input Processing

| Guard | Purpose | Action on Failure |
|-------|---------|-------------------|
| **Topic Relevance** | Ensure query is within the domain of ingested documents | Reject with explanation |
| **Input Validation** | Detect and redact sensitive data fields | Sanitize and warn user |
| **Adversarial Input** | Detect malformed or unexpected input patterns | Reject silently |
| **Content Filter** | Block invalid or malformed queries | Reject with warning |
| **Query Length** | Enforce min/max query length | Ask user to refine |

#### 3.2 Output Processing

| Guard | Purpose | Action on Failure |
|-------|---------|-------------------|
| **Accuracy Check** | Verify every claim is grounded in retrieved context | Flag ungrounded claims, re-generate |
| **Citation Validator** | Ensure all citations map to real source documents | Remove phantom citations |
| **Content Quality** | Block non-compliant or non-compliant content | Re-generate with stricter prompt |
| **Format Validator** | Ensure output matches expected schema (JSON, markdown) | Re-format |
| **Factual Consistency** | Cross-check numerical claims and dates | Flag inconsistencies |

#### 3.3 Processing Implementation Pattern

```python
from guardrails import Guard
from guardrails.hub import (
    RestrictToTopic,
    InputValidation,
    ContentQuality,
)

input_guard = Guard().use_many(
    RestrictToTopic(valid_topics=["technology", "science", "research"], on_fail="exception"),
    InputValidation(sensitive_fields=["email", "phone", "ssn"], on_fail="fix"),
)

output_guard = Guard().use_many(
    AccuracyCheck(threshold=0.7, on_fail="reask"),
    ContentQuality(threshold=0.8, on_fail="filter"),
)
```

#### 3.4 Evaluation Pipeline (Ragas + DeepEval)

**Golden Dataset Creation:**
- Curate 50–100 question-answer-context triples manually
- Augment with LLM-generated synthetic Q&A pairs via Ragas `TestsetGenerator`
- Version datasets with timestamps for reproducibility

**Ragas Metrics:**

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Faithfulness** | Is the answer grounded in the retrieved context? | > 0.85 |
| **Answer Relevancy** | Does the answer address the user's question? | > 0.90 |
| **Context Precision** | Are the retrieved chunks relevant? | > 0.80 |
| **Context Recall** | Were all necessary chunks retrieved? | > 0.75 |
| **Answer Correctness** | Does the answer match the ground truth? | > 0.80 |

**DeepEval Metrics:**

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **G-Eval** | LLM-as-judge scoring on coherence, fluency, relevance | > 4.0/5.0 |
| **Hallucination** | Percentage of claims not supported by context | < 0.15 |
| **Bias** | Detection of gender, racial, political bias | < 0.10 |
| **Summarization** | Quality of condensed answers | > 0.80 |

**Evaluation Execution:**

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

results = evaluate(
    dataset=golden_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=eval_llm,
    embeddings=eval_embeddings,
)
```

**CI/CD Integration:**
- Run evals on every PR / model change
- Fail pipeline if metrics drop below thresholds
- Generate comparison reports between model versions

---

### Phase 4: Fine-Tuning & Self-Improvement

> **Goal:** Close the loop — use the system's own high-quality outputs to fine-tune a smaller, faster, cheaper model.

#### 4.1 Dataset Curation Pipeline
1. **Collect** all Q&A interactions from production usage
2. **Score** each interaction using the Ragas eval pipeline
3. **Filter** pairs where `faithfulness > 0.9 AND answer_relevancy > 0.9`
4. **Format** into instruction-tuning format:
   ```json
   {
     "instruction": "Answer the following research question using the provided context.",
     "input": "Context: {retrieved_chunks}\n\nQuestion: {user_query}",
     "output": "{high_quality_answer}"
   }
   ```
5. **Deduplicate** and **split** into train/val/test (80/10/10)

#### 4.2 Fine-Tuning Configuration

**Base Model:** Mistral 7B Instruct v0.3 (or Llama 3.1 8B Instruct)

**Method:** QLoRA (4-bit quantization + LoRA adapters)

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    max_seq_length=4096,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    max_seq_length=4096,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    output_dir="./checkpoints",
)
```

#### 4.3 A/B Evaluation
- Run the same eval dataset through both the base model and fine-tuned model
- Generate a comparison report with metric-by-metric deltas
- Deploy the fine-tuned model only if it improves metrics across the board

#### 4.4 Deployment
- Export to GGUF for Ollama local serving
- Or deploy to HuggingFace Inference Endpoints
- Update the system to use the fine-tuned model as default, with fallback to base model

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.11+ | Core language |
| **Package Manager** | `uv` or `poetry` | Dependency management |
| **LLM Framework** | LangChain 0.3+ | Chains, retrievers, document loaders |
| **Agent Orchestration** | LangGraph 0.2+ | Stateful multi-agent graphs |
| **Agent Framework** | CrewAI 0.80+ | Role-based agent definitions |
| **Vector Store (Dev)** | ChromaDB 0.5+ | Local persistent vector store |
| **Vector Store (Prod)** | Pinecone (Serverless) | Managed cloud vector store |
| **Embeddings** | OpenAI / HuggingFace | Document and query embeddings |
| **Re-ranking** | `cross-encoder` / Cohere Rerank | Result re-ranking |
| **Guardrails** | Guardrails AI 0.5+ | Input/output processing |
| **Evaluation** | Ragas 0.2+ / DeepEval 1.0+ | RAG quality benchmarking |
| **Fine-Tuning** | Unsloth / HuggingFace TRL | QLoRA fine-tuning |
| **LLM Providers** | OpenAI, Anthropic, Ollama | Generation models |
| **UI Framework** | Streamlit / Chainlit | Chat interface |
| **API Framework** | FastAPI | REST API endpoints |
| **Observability** | LangSmith / Arize Phoenix | Tracing, debugging, monitoring |
| **Database** | SQLite / PostgreSQL | Metadata, user sessions, eval results |
| **Task Queue** | Celery + Redis (optional) | Async document ingestion |
| **Containerization** | Docker + Docker Compose | Reproducible deployments |
| **CI/CD** | GitHub Actions | Automated testing and eval |

---

## Data Flow & Pipeline Architecture

### Document Ingestion Flow

```
User uploads documents
        │
        ▼
┌─────────────────┐
│  File Validator  │  ← Verify format, size limits, file type check
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Document Loader │  ← LangChain loader (type-specific)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Metadata        │  ← Extract: source, title, author, date, page
│  Extractor       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Cleaner    │  ← Remove headers/footers, normalize whitespace
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Chunking Engine │  ← Split into overlapping semantic chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Deduplication   │  ← Content-hash check against existing chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embedding       │  ← Batch embed all new chunks
│  Pipeline        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store    │  ← Upsert embeddings + metadata
│  (Chroma/Pine)   │
└─────────────────┘
```

### Query Processing Flow

```
User Query
    │
    ▼
┌──────────────────┐
│  INPUT PROCESSING │  ← Validation, unexpected detection, topic, length
└────────┬─────────┘
         │ (pass)
         ▼
┌──────────────────┐
│  QUERY ANALYST    │  ← Decompose, refine, extract filters
│  (Agent 1)        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  RETRIEVAL        │  ← Multi-query + hybrid search + re-rank
│  (Agent 2 tool)   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  RESEARCHER       │  ← Analyze chunks, extract key findings
│  (Agent 2)        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  FACT-CHECKER     │  ← Cross-verify claims against sources
│  (Agent 3)        │
└────────┬─────────┘
         │
    ┌────┴────┐
    │ PASS?   │
    └────┬────┘
    YES  │  NO → Loop back to RESEARCHER with feedback
         ▼
┌──────────────────┐
│  SYNTHESIZER      │  ← Combine into coherent answer with citations
│  (Agent 4)        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  CRITIC           │  ← Evaluate quality, completeness, clarity
│  (Agent 5)        │
└────────┬─────────┘
         │
    ┌────┴────┐
    │ PASS?   │
    └────┬────┘
    YES  │  NO → Loop back to SYNTHESIZER with critique (max 3x)
         ▼
┌──────────────────┐
│  OUTPUT PROCESSING│  ← Accuracy, content quality, citation validity
└────────┬─────────┘
         │ (pass)
         ▼
┌──────────────────┐
│  DELIVER          │  ← Stream answer to user with citations
└──────────────────┘
```

---

## Vector Store Strategy

### ChromaDB (Development)

```python
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./data/chroma_db",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True,
    )
)

collection = client.get_or_create_collection(
    name="research_documents",
    metadata={"hnsw:space": "cosine"},
    embedding_function=openai_ef,  # or huggingface_ef
)

# Upsert with metadata
collection.upsert(
    ids=[chunk.id for chunk in chunks],
    documents=[chunk.text for chunk in chunks],
    metadatas=[chunk.metadata for chunk in chunks],
    embeddings=[chunk.embedding for chunk in chunks],
)

# Query with metadata filtering
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10,
    where={"source_type": "pdf", "date": {"$gte": "2024-01-01"}},
    include=["documents", "metadatas", "distances"],
)
```

### Pinecone (Production)

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index = pc.Index("deepresearch")

# Upsert with namespace isolation
index.upsert(
    vectors=[
        {
            "id": chunk.id,
            "values": chunk.embedding,
            "metadata": {
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page,
                "date": chunk.date,
            }
        }
        for chunk in chunks
    ],
    namespace="user-123",  # Multi-tenant isolation
)

# Hybrid search
results = index.query(
    vector=query_embedding,
    sparse_vector=sparse_embedding,  # BM25/SPLADE
    top_k=10,
    filter={"source_type": {"$eq": "pdf"}},
    include_metadata=True,
    namespace="user-123",
)
```

---

## API & UI Specification

### FastAPI Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/ingest` | Upload and ingest documents |
| `GET` | `/api/v1/collections` | List all document collections |
| `DELETE` | `/api/v1/collections/{id}` | Delete a collection |
| `POST` | `/api/v1/query` | Submit a research query |
| `GET` | `/api/v1/query/{id}/stream` | SSE stream for query response |
| `GET` | `/api/v1/evals/run` | Trigger evaluation pipeline |
| `GET` | `/api/v1/evals/results` | Get evaluation results |
| `GET` | `/api/v1/health` | Health check |

### Streamlit UI Pages

1. **📚 Knowledge Base** — Upload documents, view collections, manage ingested content
2. **💬 Research Chat** — Chat interface with streaming answers, citations sidebar, agent activity log
3. **📊 Eval Dashboard** — Metrics visualization, historical trends, model comparison charts
4. **⚙️ Settings** — LLM provider config, chunking params, processing toggles

---

## Directory Structure

```
deepresearch-ai/
├── .env.example                    # Environment variables template
├── .gitignore
├── README.md
├── pyproject.toml                  # Project config (uv/poetry)
├── Dockerfile
├── docker-compose.yml
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py             # Pydantic settings (env vars)
│   │   └── constants.py            # Magic numbers, defaults
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loaders.py              # Multi-format document loaders
│   │   ├── chunker.py              # Chunking strategies
│   │   ├── embedder.py             # Embedding pipeline
│   │   ├── deduplicator.py         # Content-hash dedup
│   │   └── pipeline.py             # End-to-end ingestion orchestrator
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract VectorStore interface
│   │   ├── chroma_store.py         # ChromaDB implementation
│   │   ├── pinecone_store.py       # Pinecone implementation
│   │   └── manager.py              # Factory + connection manager
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── dense.py                # Dense vector retrieval
│   │   ├── sparse.py               # BM25 sparse retrieval
│   │   ├── hybrid.py               # RRF fusion
│   │   ├── reranker.py             # Cross-encoder re-ranking
│   │   ├── multi_query.py          # Multi-query expansion
│   │   └── pipeline.py             # Full retrieval pipeline
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── state.py                # LangGraph state definition
│   │   ├── graph.py                # LangGraph graph definition
│   │   ├── nodes.py                # Graph node implementations
│   │   ├── crew.py                 # CrewAI agent/crew definitions
│   │   ├── tools.py                # Agent tools (search, web)
│   │   └── prompts.py              # Agent system prompts
│   │
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── input_guards.py         # Input processing
│   │   ├── output_guards.py        # Output processing
│   │   └── validators.py           # Custom validator implementations
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── datasets.py             # Golden dataset management
│   │   ├── ragas_eval.py           # Ragas evaluation runner
│   │   ├── deepeval_eval.py        # DeepEval evaluation runner
│   │   ├── reports.py              # Eval report generation
│   │   └── synthetic.py            # Synthetic test data generation
│   │
│   ├── finetuning/
│   │   ├── __init__.py
│   │   ├── curator.py              # Q&A pair curation pipeline
│   │   ├── formatter.py            # Dataset formatting (Alpaca/ShareGPT)
│   │   ├── trainer.py              # QLoRA training script
│   │   ├── evaluator.py            # A/B model comparison
│   │   └── exporter.py             # GGUF export for Ollama
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app
│   │   ├── routes/
│   │   │   ├── ingest.py
│   │   │   ├── query.py
│   │   │   ├── evals.py
│   │   │   └── health.py
│   │   └── middleware.py           # CORS, rate limiting, auth
│   │
│   └── ui/
│       ├── app.py                  # Streamlit main app
│       ├── pages/
│       │   ├── knowledge_base.py
│       │   ├── research_chat.py
│       │   ├── eval_dashboard.py
│       │   └── settings.py
│       └── components/
│           ├── chat.py
│           ├── file_upload.py
│           └── metrics_chart.py
│
├── data/
│   ├── chroma_db/                  # ChromaDB persistent storage
│   ├── golden_datasets/            # Eval datasets (versioned)
│   ├── finetuning/                 # Fine-tuning datasets
│   └── sample_docs/                # Sample documents for demo
│
├── tests/
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_retrieval.py
│   │   ├── test_guardrails.py
│   │   └── test_agents.py
│   ├── integration/
│   │   ├── test_ingestion_pipeline.py
│   │   ├── test_query_pipeline.py
│   │   └── test_eval_pipeline.py
│   └── conftest.py                 # Shared fixtures
│
├── scripts/
│   ├── ingest_sample_data.py       # Quick-start data loading
│   ├── run_evals.py                # Standalone eval runner
│   ├── generate_synthetic.py       # Synthetic dataset generator
│   └── export_model.py             # Model export script
│
├── notebooks/
│   ├── 01_rag_exploration.ipynb    # RAG experimentation
│   ├── 02_agent_debugging.ipynb    # Agent flow debugging
│   ├── 03_eval_analysis.ipynb      # Eval results analysis
│   └── 04_finetuning.ipynb         # Fine-tuning experiments
│
└── docs/
    ├── architecture.md
    ├── setup.md
    ├── api_reference.md
    └── evaluation_guide.md
```

---

## 🎯 Development Prompt

> **Copy and use this prompt to guide the AI-assisted development of this project. It is designed to be self-contained and comprehensive.**

---

### PROMPT: Build DeepResearch AI — Multi-Agent Research Platform

```
You are building "DeepResearch AI", a production-grade multi-agent research
platform in Python. The project is built in 4 incremental phases, each
producing a working system. Follow these instructions precisely.

═══════════════════════════════════════════════════════════════════════
PHASE 1: FOUNDATION RAG PIPELINE
═══════════════════════════════════════════════════════════════════════

Build a complete document ingestion and retrieval pipeline:

1. DOCUMENT INGESTION
   - Create multi-format document loaders using LangChain:
     • PyPDFLoader for PDFs
     • WebBaseLoader for web pages
     • UnstructuredMarkdownLoader for markdown files
     • CSVLoader for structured data
   - Extract metadata: filename, source, page number, ingestion timestamp
   - Implement content-hash deduplication to prevent re-indexing

2. CHUNKING ENGINE
   - Implement RecursiveCharacterTextSplitter with:
     • chunk_size=512 tokens
     • chunk_overlap=64 tokens
     • Hierarchical separators: ["\n\n", "\n", ". ", " "]
   - Also implement SemanticChunker as an alternative strategy
   - Support parent-child document relationships

3. EMBEDDING PIPELINE
   - Primary: OpenAI text-embedding-3-small (1536 dimensions)
   - Fallback: HuggingFace BAAI/bge-small-en-v1.5 (local)
   - Batch processing with rate limiting
   - Embedding cache to avoid recomputation

4. VECTOR STORE (PERSISTENT)
   - ChromaDB in persistent mode (SQLite backend) for local development:
     • Create collections with cosine similarity HNSW index
     • Support metadata filtering on upsert and query
   - Pinecone serverless for production:
     • Namespace-based multi-tenant isolation
     • Hybrid search support (dense + sparse)
   - Create a VectorStoreManager abstraction that switches via config

5. RETRIEVAL ENGINE
   - Dense retrieval via vector store cosine similarity
   - Sparse retrieval via BM25 (rank_bm25 library)
   - Reciprocal Rank Fusion (RRF) to combine dense + sparse
   - Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2)
   - Multi-query retrieval: generate 3 query variations for recall
   - Self-query: extract metadata filters from natural language

6. GENERATION
   - Support OpenAI, Anthropic, and Ollama providers
   - SSE streaming for real-time token delivery
   - Grounded system prompt: cite sources, refuse when context
     is insufficient
   - ConversationBufferWindowMemory (window=5)

7. USER INTERFACE
   - Streamlit app with:
     • File upload sidebar for document ingestion
     • Chat interface with streaming responses
     • Source citations displayed below each answer
     • Collection management (list, delete)

═══════════════════════════════════════════════════════════════════════
PHASE 2: MULTI-AGENT ORCHESTRATION
═══════════════════════════════════════════════════════════════════════

Replace the single RAG chain with a multi-agent system:

1. LANGGRAPH STATE MACHINE
   - Define ResearchState TypedDict with fields:
     query, refined_query, sub_queries, retrieved_contexts,
     research_notes, fact_check_report, fact_check_passed,
     synthesis, critique, critique_passed, citations,
     final_answer, iteration_count, max_iterations
   - Build a directed graph with nodes:
     INTAKE → DECOMPOSE → RESEARCH → FACT_CHECK → SYNTHESIZE → CRITIQUE → DELIVER
   - Add conditional edges:
     • FACT_CHECK → RESEARCH (if failed, with refined query)
     • CRITIQUE → SYNTHESIZE (if failed, with feedback, max 3 loops)
   - Enable LangGraph checkpointing for session persistence

2. CREWAI AGENTS
   Define 5 specialized agents:

   a) Query Analyst: decompose complex queries into sub-questions,
      classify intent, extract metadata filters
   b) Researcher: search vector store via multi-query retrieval,
      extract key findings, summarize relevant chunks
   c) Fact-Checker: independently verify claims against source
      documents, detect contradictions, score alignment
   d) Synthesizer: combine verified findings into coherent answer
      with proper citations in [Source: filename, page X] format
   e) Critic: evaluate answer for completeness, clarity, accuracy;
      provide actionable revision feedback

3. CREW CONFIGURATION
   - Process: Sequential (or hierarchical with manager agent)
   - Enable shared memory and caching
   - Set max_rpm=30 for rate limiting
   - Verbose logging for debugging

4. UI UPDATES
   - Add agent activity log panel showing which agent is active
   - Show intermediate steps (research notes, fact-check results)
   - Display iteration count for critique loops

═══════════════════════════════════════════════════════════════════════
PHASE 3: GUARDRAILS & EVALUATION
═══════════════════════════════════════════════════════════════════════

1. INPUT GUARDRAILS (Guardrails AI)
   - TopicRelevance: reject queries outside ingested document domains
   - Input Validation: detect and redact sensitive data fields
   - Adversarial Input Detection: detect malformed or unexpected input patterns
   - Content Filter: reject invalid or malformed queries
   - Query Length: enforce 10-500 character range

2. OUTPUT GUARDRAILS (Guardrails AI)
   - Accuracy Check: verify claims against retrieved context
     using ProvenanceEmbeddings (threshold=0.7)
   - Citation Validator: ensure all citations map to real sources
   - Content Quality: block non-compliant or non-compliant outputs
   - Format Validator: ensure markdown/JSON schema compliance

3. EVALUATION PIPELINE
   a) Golden Dataset:
      - Manually curate 50 question-answer-context triples
      - Generate 50 more via Ragas TestsetGenerator
      - Version with timestamps

   b) Ragas Metrics (targets):
      - Faithfulness > 0.85
      - Answer Relevancy > 0.90
      - Context Precision > 0.80
      - Context Recall > 0.75

   c) DeepEval Metrics (targets):
      - G-Eval coherence > 4.0/5.0
      - Hallucination rate < 0.15
      - Bias score < 0.10

   d) Execution:
      - CLI script: python scripts/run_evals.py
      - Generate HTML report with charts
      - Compare against previous runs

4. UI UPDATES
   - Add Eval Dashboard page with metric visualizations
   - Historical trend charts (line graphs over time)
   - Model comparison table

═══════════════════════════════════════════════════════════════════════
PHASE 4: FINE-TUNING & SELF-IMPROVEMENT
═══════════════════════════════════════════════════════════════════════

1. DATASET CURATION
   - Collect all Q&A interactions from the system
   - Score each using Ragas faithfulness + relevancy
   - Filter pairs where both scores > 0.9
   - Format into instruction-tuning format (Alpaca style):
     {"instruction": "...", "input": "context + question", "output": "answer"}
   - Deduplicate and split 80/10/10 train/val/test

2. FINE-TUNING (QLoRA)
   - Base model: Mistral 7B Instruct v0.3
   - Method: 4-bit quantization + LoRA (r=16, alpha=16)
   - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj,
     up_proj, down_proj
   - Training: 3 epochs, lr=2e-4, batch_size=4, grad_accum=4
   - Use Unsloth for 2x training speed

3. A/B EVALUATION
   - Run eval dataset through both base and fine-tuned model
   - Generate comparison report with metric deltas
   - Deploy fine-tuned only if ALL metrics improve

4. DEPLOYMENT
   - Export to GGUF format for Ollama serving
   - Or deploy to HuggingFace Inference Endpoints
   - Update system config to use fine-tuned as default

═══════════════════════════════════════════════════════════════════════
TECHNICAL REQUIREMENTS
═══════════════════════════════════════════════════════════════════════

- Python 3.11+, managed with uv or poetry
- Type hints on all functions
- Pydantic settings for configuration
- Comprehensive docstrings
- Unit tests for core modules (pytest)
- Integration tests for pipelines
- Docker + docker-compose for deployment
- GitHub Actions CI: lint + test + eval
- .env.example with all required environment variables
- README with setup instructions, architecture diagram, and demo GIF

═══════════════════════════════════════════════════════════════════════
CODING STANDARDS
═══════════════════════════════════════════════════════════════════════

- Use async/await for I/O-bound operations
- Implement proper error handling with custom exceptions
- Use structured logging (structlog or loguru)
- Follow the Repository pattern for data access
- Use dependency injection for testability
- Keep functions under 50 lines
- No hardcoded secrets — use environment variables
- All LLM calls must have timeout and retry logic
```

---

## README Template

Below is a starter README for the project repository:

```markdown
# 🧠 DeepResearch AI

**Autonomous Multi-Agent Research Platform with Persistent Knowledge Base**

> Ingest documents → Build searchable knowledge → Query with a team of
> AI agents → Get evaluated, cited answers

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)]()
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)]()
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple.svg)]()
[![CrewAI](https://img.shields.io/badge/CrewAI-0.80-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

## Quick Start

\```bash
# Clone
git clone https://github.com/yourusername/deepresearch-ai.git
cd deepresearch-ai

# Install dependencies
pip install uv
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Ingest sample documents
python scripts/ingest_sample_data.py

# Launch the app
streamlit run src/ui/app.py
\```

## Architecture

[See full architecture diagram and documentation in docs/architecture.md]

## Evaluation Results

| Metric | Score | Target |
|--------|-------|--------|
| Faithfulness | 0.91 | > 0.85 |
| Answer Relevancy | 0.93 | > 0.90 |
| Context Precision | 0.87 | > 0.80 |
| Hallucination Rate | 0.08 | < 0.15 |

## License

MIT
```

---

> [!TIP]
> **Getting Started:** Create a new repository, copy the development prompt into your AI coding assistant, and start with Phase 1. Each phase builds on the previous one and produces a working system you can demo at any point.
