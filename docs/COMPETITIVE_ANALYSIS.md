# Competitive Analysis — Deep Research Projects

**Date:** 2026-09-15
**Purpose:** Understand the landscape and differentiate DeepResearch AI

---

## Executive Summary

There are 6+ major "deep research" projects on GitHub. Most are **web search wrappers** — they search the internet, scrape pages, and summarize with LLMs. **DeepResearch AI is fundamentally different**: it's a **document research platform** with persistent knowledge bases, multi-agent orchestration, guardrails, evaluation, and fine-tuning.

**Key Insight:** We're not competing with web search wrappers. We're building something they don't have.

---

## Competitive Landscape

### 1. dzhng/deep-research (19.7k stars)

| Aspect | Details |
|--------|---------|
| **Language** | TypeScript/Node.js |
| **Lines of Code** | ~500 |
| **Purpose** | Web research assistant |
| **Input** | User query → searches the web |
| **Output** | Markdown report with sources |
| **Architecture** | Simple recursive agent |
| **Vector Store** | None |
| **RAG** | None |
| **Multi-Agent** | None |
| **Guardrails** | None |
| **Evaluation** | None |
| **Fine-tuning** | None |

**How it works:**
1. User asks a question
2. Agent generates search queries
3. Firecrawl searches the web
4. LLM extracts learnings from results
5. If depth > 0, recursively research deeper
6. Generate final markdown report

**Verdict:** Simple, elegant, but NOT a production-grade research platform. Good for web search, not for document-based research.

---

### 2. langchain-ai/open_deep_research (12.6k stars)

| Aspect | Details |
|--------|---------|
| **Language** | Python |
| **Purpose** | Web research agent |
| **Architecture** | LangGraph state machine |
| **Vector Store** | None (web search only) |
| **RAG** | None |
| **Multi-Agent** | ✅ LangGraph |
| **Guardrails** | None |
| **Evaluation** | ✅ Deep Research Bench |
| **Fine-tuning** | None |

**Key Features:**
- Official LangChain project
- Uses LangGraph for orchestration
- Has evaluation with Deep Research Bench (100 PhD-level tasks)
- Supports multiple LLM providers (OpenAI, Anthropic, etc.)

**Verdict:** More sophisticated than dzhng/deep-research, but still focused on web search. No document ingestion, no persistent knowledge base.

---

### 3. Alibaba-NLP/DeepResearch (19.9k stars)

| Aspect | Details |
|--------|---------|
| **Language** | Python |
| **Purpose** | Web research agent |
| **Architecture** | ReAct + IterResearch |
| **Vector Store** | FAISS (local) |
| **RAG** | ✅ Local documents |
| **Multi-Agent** | ✅ |
| **Guardrails** | None |
| **Evaluation** | ✅ Benchmark datasets |
| **Fine-tuning** | ✅ (Tongyi-DeepResearch-30B-A3B) |

**Key Features:**
- Leading open-source deep research agent
- Has own fine-tuned model (30B parameters)
- Supports document processing via file parser
- Has benchmark evaluation scripts

**Verdict:** The most comprehensive competitor. Has RAG and fine-tuning, but focused on their own model, not production-grade orchestration stack.

---

### 4. tarun7r/deep-research-agent (182 stars)

| Aspect | Details |
|--------|---------|
| **Language** | Python |
| **Purpose** | Web research agent |
| **Architecture** | LangGraph + LangChain |
| **Vector Store** | None |
| **RAG** | None |
| **Multi-Agent** | ✅ 4 agents (Planner, Searcher, Synthesizer, Writer) |
| **Guardrails** | None |
| **Evaluation** | None |
| **Fine-tuning** | None |

**Key Features:**
- Multi-agent with LangGraph
- Credibility scoring for sources
- Quality validation with retry logic
- LLM usage tracking

**Verdict:** Similar multi-agent approach to ours, but no vector store, no guardrails, no evaluation, no fine-tuning.

---

### 5. crewAIInc/template_deep_research (20 stars)

| Aspect | Details |
|--------|---------|
| **Language** | Python |
| **Purpose** | Web research template |
| **Architecture** | CrewAI Flow |
| **Vector Store** | None |
| **RAG** | None |
| **Multi-Agent** | ✅ CrewAI |
| **Guardrails** | None |
| **Evaluation** | None |
| **Fine-tuning** | None |

**Key Features:**
- Official CrewAI template
- Simple router pattern (casual chat vs search)
- Uses Firecrawl for web search
- Streamlit frontend

**Verdict:** Template for learning CrewAI, not a production system.

---

### 6. syedmunimshah/deep-research-agent-system (0 stars)

| Aspect | Details |
|--------|---------|
| **Language** | Python |
| **Purpose** | Web research agent |
| **Architecture** | LangGraph |
| **Vector Store** | FAISS |
| **RAG** | ✅ Local documents |
| **Multi-Agent** | ✅ (Planner, Researcher, Critic, Synthesizer) |
| **Guardrails** | None |
| **Evaluation** | None |
| **Fine-tuning** | None |

**Key Features:**
- Self-correcting reasoning loops
- Long-horizon memory (SQLite)
- LangGraph checkpointing
- Local RAG with FAISS

**Verdict:** Closest to our architecture, but no guardrails, no evaluation, no fine-tuning, no production UI.

---

## Comparison Matrix

| Feature | dzhng | langchain | Alibaba | tarun7r | crewAI | syedmunim | **DeepResearch AI** |
|---------|-------|-----------|---------|---------|--------|-----------|---------------------|
| **Language** | TS | Python | Python | Python | Python | Python | **Python** |
| **Web Search** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Document Ingestion** | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | **✅** |
| **Persistent Vector Store** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| **RAG Pipeline** | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | **✅** |
| **Multi-Agent** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | **✅** |
| **LangGraph** | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | **✅** |
| **CrewAI** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | **✅** |
| **Guardrails** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| **Evaluation** | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | **✅** |
| **Fine-tuning** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | **✅** |
| **Production UI** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | **✅** |
| **FastAPI** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |

---

## Our Differentiation

### What We Have That Others Don't

1. **Persistent Knowledge Base**
   - Others: Search the web each time
   - We: Build a persistent vector store from uploaded documents

2. **Document Ingestion Pipeline**
   - Others: No document handling (except Alibaba)
   - We: Multi-format loaders, chunking, embedding, deduplication

3. **Full Production Stack**
   - Others: LangGraph OR CrewAI
   - We: LangGraph + CrewAI + Guardrails + Ragas + DeepEval + Fine-tuning

4. **Evaluation-Driven**
   - Others: No quality metrics (except langchain and Alibaba)
   - We: Faithfulness, relevancy, context precision, hallucination detection

5. **Self-Improvement Loop**
   - Others: Static
   - We: Fine-tune on high-scoring Q&A pairs

6. **Guardrails/Safety**
   - Others: None
   - We: Input validation, output processing, PII detection

### Positioning Statement

**Don't call it "deep research"** — that's a crowded category of web search wrappers.

**Call it:** "Production-Grade Multi-Agent Research Platform with Persistent Knowledge Base"

### Resume Impact

**Before:** "Built a deep research agent"

**After:** "Built a production-grade multi-agent research platform that ingests documents into a persistent vector store, orchestrates specialized AI agents via LangGraph, validates outputs with Guardrails AI, and evaluates quality with Ragas/DeepEval — the full stack companies expect for production AI systems."

---

## Lessons From Competitors

### 1. Presentation Matters
- dzhng/deep-research has 19.7k stars because its README is clean and its flow diagram is clear
- **Action:** Copy their presentation style for our README

### 2. Evaluation Is Valuable
- langchain-ai/open_deep_research has evaluation with Deep Research Bench
- **Action:** Reference their approach for our Ragas/DeepEval integration

### 3. Simplicity Sells
- dzhng/deep-research is only 500 lines of code
- **Action:** Keep each component at "interview-discussable" depth

### 4. Multi-Agent Is Expected
- 5/6 projects use multi-agent orchestration
- **Action:** Our LangGraph + CrewAI approach is industry-standard

---

## Strategy

### What NOT to Build
- Web search integration (not our focus)
- Custom LLM training (use existing models)
- Complex web scraping (use document loaders)

### What TO Build
1. **LangGraph orchestration** — converts resume claim to evidence
2. **CrewAI delegation** — demonstrates agent collaboration
3. **Guardrails AI** — proves safety engineering
4. **Ragas + DeepEval** — validates quality
5. **Fine-tuning pipeline** — shows MLOps skills

### 4-Week Timeline
- **Week 1:** LangGraph refactor
- **Week 2:** Guardrails AI (3 guards)
- **Week 3:** CrewAI (1 agent)
- **Week 4:** Ragas + DeepEval integration

---

*This document should be updated as new competitors emerge or our strategy evolves.*
