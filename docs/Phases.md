# Implementation Phases

## Revised Strategy (ADR-004)

**Context:** User is pivoting from Java Full-Stack to AI Engineer. Goal is to showcase production-grade AI skills (LangGraph, CrewAI, Guardrails AI, Ragas, DeepEval) on resume to get shortlisted.

**Decision:** Build 5 frameworks at "interview-discussable" depth in 4 weeks. Each component must be defensible in a 45-minute technical interview.

**Reference:** See docs/COMPETITIVE_ANALYSIS.md for landscape analysis and docs/decisions.md ADR-004 for full rationale.

---

## Phase 1: Foundation RAG Pipeline
**Status:** COMPLETE
**Goal:** Build a robust, production-quality RAG pipeline with a persistent vector store.

### Completed Tasks
- [x] 1.1 Document Ingestion - loaders (PDF, web, MD, CSV), metadata extraction, dedup
- [x] 1.2 Chunking Engine - recursive + semantic strategies
- [x] 1.3 Embedding Pipeline - OpenAI + HuggingFace fallback, batch processing
- [x] 1.4 Vector Store - ChromaDB persistent + Pinecone stub, VectorStoreManager
- [x] 1.5 Retrieval Engine - dense, sparse, hybrid (RRF), reranker, multi-query
- [x] 1.6 Generation Pipeline - LLM providers, streaming, conversation memory
- [x] 1.7 API Layer - FastAPI endpoints (ingest, query, health)
- [x] 1.8 UI Layer - Streamlit chat + file upload + citations

### Definition of Done
- [x] All ingestion loaders work with test documents
- [x] ChromaDB upsert-then-search round-trip succeeds
- [x] RRF fusion tests pass with known inputs
- [x] API health endpoint returns 200
- [x] Streamlit app launches and can answer a query with citations
- [x] All unit tests pass
- [x] Lint clean

**Result:** 41 tests passing, full pipeline working end-to-end.

---

## Phase 2: Multi-Agent Orchestration (Revised)
**Status:** IN PROGRESS
**Goal:** Build LangGraph + CrewAI + Guardrails + Eval at interview-discussable depth in 4 weeks.

### Week 1: LangGraph Integration (Days 1-2)
**Priority:** HIGH - Converts resume claim to code evidence

**Tasks:**
- [ ] 2.1.1 Add langgraph to pyproject.toml
- [ ] 2.1.2 Rewrite src/agents/graph.py using StateGraph
- [ ] 2.1.3 Keep existing node logic unchanged (self_critique.py)
- [ ] 2.1.4 Add conditional edges for pass/fail routing
- [ ] 2.1.5 Implement loop guard (max 3 iterations)
- [ ] 2.1.6 Run existing tests to confirm nothing breaks

**Definition of Done:**
- [ ] Graph compiles with LangGraph StateGraph
- [ ] Node logic unchanged (self_critique.py works)
- [ ] Conditional edges route correctly (pass/fail)
- [ ] Loop guard prevents infinite cycles
- [ ] All existing tests pass

**Files to Modify:**
- pyproject.toml - add langgraph dependency
- src/agents/graph.py - rewrite using StateGraph

**Interview Talking Points:**
- "I chose LangGraph over a raw state machine because it provides persistence, visualization, and conditional edges out of the box"
- "The graph has 5 states: INTAKE, RESEARCH, CRITIQUE, REVISE, DELIVER"
- "Loop guard prevents infinite critique cycles with max_iterations parameter"

---

### Week 2: Guardrails AI (Days 3-5)
**Priority:** HIGH - Proves safety engineering

**Tasks:**
- [ ] 2.2.1 Add guardrails-ai to pyproject.toml
- [ ] 2.2.2 Implement InputGuard in src/guardrails/input_guards.py
  - Injection detection
  - Topic relevance check
  - Query length validation
- [ ] 2.2.3 Implement OutputGuard in src/guardrails/output_guards.py
  - Structured response format
  - Hallucination check (context grounding)
  - Citation validation
- [ ] 2.2.4 Implement HallucinationGuard in src/guardrails/validators.py
  - Compare claims against retrieved context
  - Score grounding (0-1)
  - Reject if below threshold
- [ ] 2.2.5 Write unit tests for each guard (pass AND fail cases)
- [ ] 2.2.6 Wire guards into LangGraph graph as middleware

**Definition of Done:**
- [ ] 3 guards implemented and tested
- [ ] Each guard has pass AND fail test cases
- [ ] Guards wired into graph as middleware
- [ ] 90%+ test coverage for guardrails module

**Files to Create/Modify:**
- src/guardrails/input_guards.py - input validation
- src/guardrails/output_guards.py - output processing
- src/guardrails/validators.py - custom validators
- tests/unit/test_guards.py - unit tests

**Interview Talking Points:**
- "I implemented 3 guards using Guardrails AI: input validation, output format, and hallucination detection"
- "The hallucination guard scores how well claims are grounded in retrieved context"
- "Guards are middleware, not inline - they wrap agent calls without modifying agent logic"

---

### Week 3: CrewAI Integration (Days 6-8)
**Priority:** MEDIUM - Demonstrates agent delegation

**Tasks:**
- [ ] 2.3.1 Add crewai to pyproject.toml
- [ ] 2.3.2 Create src/agents/crew.py with single CriticAgent
  - Role: Review main agent's output for quality
  - Tools: None (pure evaluation)
  - Goal: Provide actionable revision feedback
- [ ] 2.3.3 Wire CriticAgent into LangGraph graph as a node
- [ ] 2.3.4 Write tests for agent delegation
- [ ] 2.3.5 Update prompts for agent roles

**Definition of Done:**
- [ ] CriticAgent created with CrewAI
- [ ] Agent wired into LangGraph graph
- [ ] Agent reviews main output and provides feedback
- [ ] Tests pass for agent delegation

**Files to Create/Modify:**
- src/agents/crew.py - CrewAI agent definitions
- src/agents/graph.py - add CriticAgent node
- tests/unit/test_crew.py - agent tests

**Interview Talking Points:**
- "I used CrewAI for agent delegation because it provides role definitions, tool delegation, and inter-agent communication"
- "The CriticAgent reviews the main agent's output and provides actionable revision feedback"
- "I started with ONE agent to prove the pattern works before adding more"

**Note:** We are building ONE agent, not four. This demonstrates the pattern without over-engineering.

---

### Week 4: Ragas + DeepEval Integration (Days 9-11)
**Priority:** HIGH - Validates system quality

**Tasks:**
- [ ] 2.4.1 Wire src/evaluation/ragas_eval.py to agent pipeline
- [ ] 2.4.2 Add deepeval to pyproject.toml
- [ ] 2.4.3 Implement DeepEval hallucination metric
- [ ] 2.4.4 Run evaluation on 5 sample queries
- [ ] 2.4.5 Record baseline metrics
- [ ] 2.4.6 Generate comparison report

**Definition of Done:**
- [ ] Ragas metrics computed for agent outputs
- [ ] DeepEval hallucination metric implemented
- [ ] Baseline metrics recorded
- [ ] Comparison report generated

**Files to Modify:**
- src/evaluation/ragas_eval.py - wire to agent
- src/evaluation/deepeval_eval.py - add hallucination metric
- scripts/run_evals.py - update for agent pipeline

**Interview Talking Points:**
- "I used Ragas to measure faithfulness, relevancy, and context precision"
- "DeepEval provides hallucination detection as a comparison baseline"
- "The evaluation pipeline proves the multi-agent approach outperforms single-chain RAG"

---

### Week 4: Documentation (Days 12-14)
**Priority:** CRITICAL - Documentation IS the product

**Tasks:**
- [ ] 2.5.1 Write README.md with architecture diagram
- [ ] 2.5.2 Add ADRs to docs/decisions.md for each framework choice
- [ ] 2.5.3 Create demo script that runs full pipeline end-to-end
- [ ] 2.5.4 Clean up code, ensure all tests pass
- [ ] 2.5.5 Update resume with new skills

**Definition of Done:**
- [ ] README.md with clear architecture diagram
- [ ] ADRs for LangGraph, CrewAI, Guardrails AI, Ragas, DeepEval
- [ ] Demo script works end-to-end
- [ ] All tests pass
- [ ] Resume updated

**Files to Create/Modify:**
- README.md - architecture, setup, demo
- docs/decisions.md - ADRs for each framework
- scripts/demo.py - end-to-end demo
- docs/ARCHITECTURE.md - update with new components

**Interview Talking Points:**
- "Every architectural decision is documented in ADRs"
- "The README explains the architecture with diagrams"
- "The demo script runs the full pipeline end-to-end"

---

## Phase 3: Safety Layer and Evaluation (Expanded)
**Status:** IN PROGRESS (part of Phase 2)
**Goal:** Add input/output processing and automated quality benchmarking.

### Tasks
- [x] 3.1 Input processing (topic filtering, personal data protection, adversarial input detection, inappropriate language filtering, length constraints)
- [x] 3.2 Output processing (factual accuracy, source attribution, content safety, format validation)
- [x] 3.3 Golden dataset creation + synthetic augmentation
- [x] 3.4 Ragas evaluation pipeline
- [x] 3.5 DeepEval evaluation pipeline
- [ ] 3.6 Eval dashboard in Streamlit

### Definition of Done
- [ ] Every guard tested with pass AND fail inputs (90%+ coverage)
- [ ] Ragas metrics meet targets: faithfulness >0.85, relevancy >0.90
- [ ] scripts/run_evals.py produces HTML report
- [ ] Eval dashboard renders with real data

**Note:** Phase 3 is now part of Phase 2 (Weeks 2 and 4).

---

## Phase 4: Fine-Tuning and Self-Improvement
**Status:** DEFERRED
**Goal:** Use high-scoring Q&A pairs to fine-tune a smaller domain-specific model.

### Tasks
- [ ] 4.1 Dataset curation (collect, score, filter, format to Alpaca schema)
- [ ] 4.2 QLoRA training (Mistral 7B, Unsloth, r=16)
- [ ] 4.3 A/B evaluation (base vs fine-tuned)
- [ ] 4.4 GGUF export for Ollama

### Definition of Done
- [ ] Fine-tuned model exports to GGUF
- [ ] A/B eval shows improvement on all metrics
- [ ] System can use fine-tuned model via config change

**Note:** Phase 4 is deferred. Can be added later if resume needs more MLOps content.

---

## Timeline Summary

`
Week 1 (Days 1-2):   LangGraph integration
Week 2 (Days 3-5):   Guardrails AI (3 guards)
Week 3 (Days 6-8):   CrewAI (1 agent)
Week 4 (Days 9-14):  Ragas + DeepEval + Documentation
`

**Total:** 4 weeks, approximately 500 lines of new code, every line defensible.

---

## Success Criteria

### Resume Impact
- [ ] "LangGraph orchestration" - backed by graph.py using StateGraph
- [ ] "CrewAI delegation" - backed by crew.py with CriticAgent
- [ ] "Guardrails AI safety" - backed by 3 guards in src/guardrails/
- [ ] "Ragas/DeepEval evaluation" - backed by metrics in src/evaluation/

### Interview Readiness
- [ ] 2-minute explanation for each component
- [ ] 45-minute technical discussion capability
- [ ] Every decision justified in ADRs
- [ ] Demo script works end-to-end

### Technical Quality
- [ ] All tests pass
- [ ] Lint clean
- [ ] Type hints on all functions
- [ ] Docstrings on all public APIs

---

*This document was updated on 2026-09-15 per ADR-004 (Minimum Viable Integration).*
