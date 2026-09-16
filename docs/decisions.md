# Architecture Decision Records

---

## ADR-001: Project Structure Follows Module Ownership Map

**Date:** 2026-09-13
**Status:** Accepted
**Context:** Need clear module boundaries for multi-agent development. Multiple
AI agents will work on different modules in parallel, so ownership must be
unambiguous to prevent conflicts.
**Decision:** Follow the module ownership map in AGENTS.md Section 3. Each module
(ingestion, vectorstore, retrieval, agents, guardrails, evaluation, finetuning,
api, ui) is owned by a designated agent scope. Cross-module writes require
contract updates and a progress log entry.
**Consequences:** Enables parallel development by specialist agents. Requires
discipline to stay within scope. Cross-module interface changes must be
negotiated via docs/CONTRACTS.md.

---

## ADR-002: uv Chosen as Package Manager

**Date:** 2026-09-13
**Status:** Accepted
**Context:** Need fast, reliable Python dependency management that supports
Python 3.11+ and works cross-platform.
**Decision:** Use `uv` as the primary package manager (with poetry as fallback).
Dev dependencies use [dependency-groups] for default install via `uv sync`.
**Consequences:** Faster installs than pip/poetry. Dev tools require
`uv sync --extra dev` unless using dependency-groups. Lock file (uv.lock)
ensures reproducible installs.

---

## ADR-003: Prove Before You Build — Evaluation-First Architecture

**Date:** 2026-09-13
**Status:** Accepted
**Context:** Phase 1 (RAG pipeline) is complete with 41 passing tests. The original architecture calls for LangGraph state machines, CrewAI agents, Guardrails AI, Ragas/DeepEval evaluation, and fine-tuning (Phases 2-4). A full council deliberation was held to evaluate whether this architecture is sound or over-designed.

The council identified five load-bearing assumptions:
1. Multi-agent orchestration is necessary (vs. single-agent + self-critique)
2. LangGraph is the right state machine (vs. alternatives or custom)
3. CrewAI adds value over raw LangGraph
4. Guardrails AI is the right framework (vs. custom validators)
5. Ragas + DeepEval both provide unique value (vs. one framework)

No empirical data existed to support any of these assumptions.

**Decision:** Three architectural changes, in priority order:

1. **Build Ragas evaluation on Phase 1 BEFORE Phase 2.** Establish baseline metrics (faithfulness, relevancy, context precision, context recall) on the current single-chain pipeline. This provides the terrain map for all future architectural moves.

2. **Defer CrewAI until proven necessary.** Start Phase 2 with pure LangGraph for control flow. CrewAI's role abstraction overlaps with LangGraph's node system. Add CrewAI only if agent logic in nodes becomes unmanageable. This reduces framework coupling by ~50%.

3. **Phase 2 starts with self-critique single-agent, not multi-agent.** The simplest pattern that works: retrieve -> generate -> critique -> revise (3 LLM calls). Only graduate to full multi-agent (researcher -> fact-checker -> synthesizer -> critic) if evaluation shows the quality gap justifies the 4-5x latency and cost increase.

Additional decisions:
- `state.py` should be a Pydantic model (not TypedDict) with accumulation semantics for loop-backs
- `GuardResult.action_taken` should be an enum, not a string
- Contracts (section 4.1-4.5) need extension for cyclic flow (incremental augmentation of RetrievalResult)
- ChromaDB -> Pinecone migration path should be designed now, executed in Phase 4

**Consequences:**
- Phase 2 scope is reduced from "full multi-agent" to "single-agent + self-critique + LangGraph skeleton"
- Evaluation becomes a Phase 2 prerequisite, not Phase 3 afterthought
- CrewAI dependency is deferred, reducing version conflict risk
- Complexity budget reduced from ~17 points to ~9 points (LangGraph + Ragas only)
- Requires creating a golden evaluation dataset before multi-agent work begins

**Evidence:** Council deliberation (Aristotle, Ada Lovelace, Feynman, Socrates, Sun Tzu, Torvalds). Full deliberation log in `docs/meetings/council-2026-09-13-architecture.md`.

**Alternatives Considered:**
- Full multi-agent as originally designed — rejected as over-designed without evidence
- Custom asyncio state machine — rejected; LangGraph provides persistence and visualization
- Single evaluation framework (Ragas only) — considered but deferred; DeepEval adds G-Eval metrics that Ragas lacks
- Guardrails AI from Phase 2 — deferred; custom validators sufficient initially

---

## ADR-004: Minimum Viable Integration — Revised Strategy for Job Pivot

**Date:** 2026-09-15
**Status:** Accepted
**Supersedes:** ADR-003 (partial — re-evaluates simplification strategy)

**Context:** User provided full career context: 2+ years Java full-stack experience, pivoting to AI Engineer role. Goal is to showcase production-grade AI skills (LangGraph, CrewAI, Guardrails AI, Ragas, DeepEval) on resume to get shortlisted for AI Engineer positions.

Previous council advice (ADR-003) recommended simplifying to 1 agent + self-critique. This was optimized for engineering simplicity, not career advancement. With career objectives in focus, the strategy must change.

**Decision:** Build 5 frameworks at "interview-discussable" depth in 4 weeks:

1. **LangGraph (Week 1):** Rewrite graph.py using StateGraph — converts resume claim to code evidence
2. **Guardrails AI (Week 2):** Implement 3 guards (input validation, output format, hallucination check)
3. **CrewAI (Week 3):** Add single CriticAgent, not four agents — demonstrates delegation pattern
4. **Ragas + DeepEval (Week 4):** Wire evaluation to agent pipeline — proves system works

**Consequences:**
- Project reclassified from "production system" to "demonstration artifact"
- Optimization target is signal-per-hour, not feature count
- Every framework choice must be justified in ADRs (documentation IS the product)
- Resume converts claims to evidence: "Multi-Agent Orchestration" → LangGraph code
- Interview talking points: 45-minute technical discussion capability

**Evidence:** Council deliberation (Aristotle, Ada Lovelace, Feynman, Socrates, Sun Tzu, Torvalds). Full deliberation log in docs/meetings/council-2026-09-15-revised-strategy.md.

**Alternatives Considered:**
- Follow ADR-003 (simplify) — rejected; hurts job prospects by not demonstrating required frameworks
- Build full production system — rejected; 6+ weeks, risk of messy code, interview vulnerability
- Build everything superficially — rejected; hiring managers detect shallow framework usage

**Resume Impact:**
- Current: "LangChain, Multi-Agent Orchestration, RAG Pipelines"
- After: "LangGraph orchestration, CrewAI delegation, Guardrails AI safety, Ragas/DeepEval evaluation"
- Story: "I built a multi-agent research system, added guardrails and evaluation to prove it works"

**4-Week Timeline:**
- Day 1-2: Add langgraph, rewrite graph.py using StateGraph
- Day 3-5: Add guardrails-ai, implement 3 guards
- Day 6-8: Add crewai, create single CriticAgent
- Day 9-11: Wire ragas_eval.py, add deepeval
- Day 12-14: Write README.md, docs/decisions.md, demo script
