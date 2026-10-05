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

---

## ADR-005: Zen Free-Model Subagent Block — Hybrid Migration (Strip Pins + Selective Skills)

**Date:** 2026-09-21
**Status:** Accepted
**Context:** Around 2026-09-19 Opencode Zen began rejecting `opencode/*-free`
models when used as subagents (`mode: subagent` invoked via Task tool); they
remain usable as standalone/primary models. 8 of our `.opencode/agents/*.md`
subagents were pinned to free models (`mimo-v2.5-free`, `ling-3.0-flash-free`,
`deepseek-v4-flash-free`, `muse-spark-1.3-contributor-free`) and broke. The
`ingestion` pin (`opencode/ling-3.0-flash-free`) no longer even exists upstream
(renamed to `ling-3.0-flash-fin-free`). Council (Architect, Skeptic, Pragmatist,
Researcher) deliberated: pure skill conversion loses Task isolation, parallelism
(max 3), and per-agent `permission/*` scoping; pure disable kills the roster.
**Decision:**
1. Keeper subagents (`ingestion`, `retrieval`, `quality`, `api`, `ui`) — `model:`
   line removed, inherit primary model (`opencode/meta/muse-spark-1.3-contributor`
   from `opencode.json`). Zero orchestration change.
2. `@developer` — pin KEPT as `agentrouter/deepseek-v4-flash`. agent-router is a
   separate provider, unaffected by the Zen block; verified via
   `opencode auth list` (agent-router credential present) and `opencode models`
   (both `agentrouter/deepseek-v4-flash` and `agentrouter/glm-5.3` listed).
3. Checklist agents (`code-reviewer`, `security-auditor`, `doc-writer`) —
   migrated to Skills (`.opencode/skills/*/SKILL.md`), old agent files set to
   `disable: true`. Skills run inline via Skill tool, no separate model.
4. `orchestrator.md` / `architect.md` rosters updated with Task-vs-Skill routing.
**Consequences:** Subagent path works again on paid/Zen primary quota;
`@developer` keeps its cheap agent-router routing; review/audit/docs run in the
caller's context (slightly larger context, no child session). Model binding now
lives in one place (`opencode.json` + developer pin) instead of 12 files.
**Evidence:** Council deliberation 2026-09-21 (quick mode, 4 personas, unanimous
hybrid); `opencode models` output 2026-09-21 confirming model IDs.

---

## ADR-006: LangGraph for the Agent State Machine

**Date:** 2026-09-15
**Status:** Accepted
**Context:** Phase 2 needed an orchestration loop (intake → research → critique →
revise → deliver) with conditional edges and a loop guard. Options were a
hand-rolled state machine, raw LangChain chains, or LangGraph.
**Decision:** LangGraph `StateGraph` in `src/agents/graph.py`; the pure-Python
`graph_skeleton.py` stays as a zero-overhead alternative for the review-only path.
**Consequences:** Conditional edges and recursion limits come free; Pregel adds
startup cost, so request paths reuse compiled graphs. State is a Pydantic model
with accumulation semantics for loop-back context growth.
**Evidence:** `graph.py` compiles; loop-guard tests pass; interview story —
"persistence, visualization, and conditional edges out of the box".

## ADR-007: CrewAI, One Agent Only (CriticAgent)

**Date:** 2026-09-16
**Status:** Accepted
**Context:** ADR-004 budgets framework depth over breadth. A four-agent crew
(Researcher, Fact-Checker, Synthesizer, Critic) was the original vision.
**Decision:** A single CrewAI `CriticAgent` wired as a review step; role/goal/
backstory demonstrate delegation without multi-agent overhead. `review()` is
async via `kickoff_async()` (event-loop blocking fixed 2026-09-22).
**Consequences:** Proves the CrewAI pattern at minimum cost; adding agents later
is additive. Keyword-overlap fallback scoring is used when no LLM judge key exists.
**Evidence:** `test_crew.py` — 22 tests covering parsing, score extraction, and
pass/fail logic.

## ADR-008: Guardrails as Middleware with Custom Validators

**Date:** 2026-09-17
**Status:** Accepted
**Context:** Needed input validation, output checks, and hallucination detection
that wrap agent I/O without entangling agent logic (invariant #5).
**Decision:** Three guards (`InputGuard`, `OutputGuard`, `HallucinationGuard`)
with custom validators instead of guardrails-ai primitives — the `GuardResult`
contract maps poorly onto that API. Guards sit at graph intake/deliver nodes.
**Consequences:** Framework-independent safety layer; 28 guard tests with
pass AND fail cases; 90%+ coverage target for the module.
**Evidence:** `tests/unit/test_guards.py` all passing; wired into `graph.py`.

## ADR-009: Ragas + DeepEval with Lexical Fallback

**Date:** 2026-09-16
**Status:** Accepted
**Context:** Eval must run in CI with no LLM keys, yet production verdicts need
real judge scores. Thresholds (faithfulness >0.85, relevancy >0.90) assume an
LLM judge.
**Decision:** `RagasEvaluator` prefers the real `ragas` library and degrades to
deterministic lexical scorers (stemmed token overlap) otherwise; DeepEval mirrors
this. Fallback scores are informational against production bars — never lower the
bars to fit the fallback. 2026-09-22: citation markers stripped before
faithfulness scoring (metadata is not a claim).
**Consequences:** `scripts/run_evals.py` and `scripts/demo.py` work offline;
HTML reports always render; `evaluate_batch` is concurrent and order-preserving.
**Evidence:** `test_ragas_eval.py`, `test_deepeval.py`, `test_datasets.py` green;
demo prints `eval [lexical-fallback]` mode explicitly.

## ADR-010: Shared Review-Loop Base Class

**Date:** 2026-09-22
**Status:** Accepted
**Context:** 2026-09-16 architect review found ~80% duplication between
`self_critique.py` (411 lines) and `review_agent.py` (418 lines).
**Decision:** Extract `src/agents/review_loop.py` (`ReviewLoopBase` + structural
Protocols); subclasses keep only vocabulary (prompts, dimensions, verdict models,
state fields). Public APIs, prompts, and markers unchanged — both agent test
files pass unmodified.
**Consequences:** One place to fix loop mechanics; new review flavors are thin
subclasses. Result: base 335 lines, shells 176/181 lines.
**Evidence:** Full gate green — 150/150 tests, ruff clean, mypy strict clean.

## ADR-011: Gate Checks Safety, Pipeline Checks Relevance

**Date:** 2026-09-22
**Status:** Accepted
**Context:** The InputGuard topic allow-list hard-rejected queries it didn't
recognize, but keyword matching cannot tell a paraphrase from an off-topic
question — legitimate rewordings were being blocked before retrieval ever ran.
Meanwhile the API had no principled way to say "your corpus doesn't cover this"
(it 404'd only on truly empty results).
**Decision:**
1. `InputGuard._check_topic` is **advisory only** — it logs a warning and never
   creates a Violation. The gate enforces injection + length (safety).
2. Relevance is judged **after retrieval** by a coverage floor: top-1 dense
   cosine similarity must be >= `DEFAULT_MIN_COVERAGE` (0.25). Below the floor,
   `RetrievalPipeline` returns empty results with truthful `dense_results`
   counts.
3. Empty results short-circuit to a templated `NO_COVERAGE_RESPONSE` with
   zero LLM calls (research node sets `AgentState.no_coverage`; review/revise
   early-return; deliver skips the citation guard for the template).
**Consequences:** Paraphrased queries always reach retrieval; off-topic queries
get an honest "not covered" answer instead of a rejection or a hallucinated
response. Fused RRF scores are rank-based (~0.016) and unusable as a relevance
signal — the floor reads the dense leg only. Threshold needs real-embedding
validation (toy hash embeddings score everything ~0.7+, so the floor cannot
trip in the key-free demo).
**Evidence:** 7 new tests (5 floor + 2 short-circuit), full gate green —
158/158 tests, ruff clean, mypy strict clean.

## ADR-012: Pluggable Embedding Providers Behind a Protocol

**Date:** 2026-10-20
**Status:** Accepted
**Context:** Architecture invariant #2 claims "switching providers must require
only a config change." That held for chat (`LLMProvider` uses a provider
registry) but **not** for embeddings: `RetrievalPipeline` type-hinted the
concrete `Embedder`, so a second provider would have required editing every
caller. Adding Gemini surfaced the gap.

**Findings from the implementation:**

1. **Chat is genuinely config-only.** Google exposes an **OpenAI-compatible
   chat endpoint** (`GEMINI_OPENAI_BASE_URL`), so Gemini is reached by pointing
   the same `ChatOpenAI` class at a different `base_url`. No new code path.
2. **Embeddings are not.** That compatibility surface does not cover
   embeddings, so they require the native `langchain-google-genai` client and
   therefore a new class. The asymmetry is the finding.
3. **A structural `Protocol` needs no changes to existing code.**
   `EmbedderProtocol` is declared in `src/ingestion/embedder_base.py`, mirroring
   `vectorstore/base.py`. `Embedder` already satisfied it — verified by
   `isinstance(Embedder(), EmbedderProtocol) is True` **with zero edits**.
   `issubclass()` is unavailable on protocols with non-method members
   (`model_name` is a property), so structural conformance must be checked with
   `isinstance`.

**Decision:**

- `EmbedderProtocol` + `build_embedder()` factory in
  `src/ingestion/embedder_base.py` — the single point of provider selection,
  driven by `EMBEDDING_PROVIDER`.
- `GeminiEmbedder` (`src/ingestion/gemini_embedder.py`) as a near-mirror of
  `Embedder`: same batching, same `EmbeddingError` translation, same surface.
- Both ingestion and retrieval pipelines accept `EmbedderProtocol | None`.
- `_infer_provider` recognises `gemini*`; `_build_llm` maps it to `ChatOpenAI`
  with `base_url` override.

**Also fixed — a hermeticity defect this work exposed.** `CriticAgent()`
constructs a CrewAI `Agent`, which eagerly builds an LLM from `os.environ`.
Creating a local `.env` therefore failed **22 crew tests** with "Missing
credentials", despite those tests exercising only pure parsing logic. The suite
was never hermetic and would have broken in CI. Fixed with an autouse
session-scoped fixture in `tests/conftest.py` that neutralises provider env
vars with dummy values (construction checks presence, not validity; no test
performs a network call).

**Consequences:** Invariant #2 is now real rather than aspirational, and the
chat/embeddings asymmetry is documented rather than latent. `GeminiEmbedder`
raises at construction when no key is set, so misconfiguration fails loudly
instead of at the first query. `langchain-google-genai` added with no
conflicts (6 transitive packages, no downgrades).

**Evidence:** 18 new tests (stubbed clients, no network, no key), including
explicit protocol-conformance and factory-selection coverage. Gate green:
**176/176 tests**, ruff check + format clean, mypy strict clean (64 files).
