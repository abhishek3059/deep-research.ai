# AGENTS.md — Multi-Agent Development Protocol

> **Version:** 1.0.0 | **Project:** DeepResearch AI | **Last Updated:** _{date}_
>
> This file is the **single source of truth** for all AI coding agents working
> on this codebase. Every agent MUST read this file before making any changes.
> Sections are labeled with audience tags so agents can skip irrelevant blocks.

---

## Table of Contents

1. [Project Identity](#1-project-identity)
2. [Documentation Index](#2-documentation-index)
3. [Module Ownership Map](#3-module-ownership-map)
4. [Interface Contracts](#4-interface-contracts)
5. [Verification Protocol](#5-verification-protocol)
6. [Coding Standards](#6-coding-standards)
7. [Architecture Invariants](#7-architecture-invariants)
8. [Commands](#8-commands)
9. [Testing Requirements](#9-testing-requirements)
10. [Configuration](#10-configuration)
11. [Session Memory & Progress](#11-session-memory--progress)
12. [Multi-Agent Coordination Protocol](#12-multi-agent-coordination-protocol)
13. [Emergency Procedures](#13-emergency-procedures)
14. [Changelog](#14-changelog)

---

## 1. Project Identity

<!-- AUDIENCE: ALL AGENTS — always read this section -->

**Name:** DeepResearch AI
**Type:** Multi-agent research platform with persistent knowledge base
**Language:** Python 3.11+
**Package Manager:** uv (preferred) or poetry
**Key Frameworks:** LangChain, LangGraph, CrewAI, Guardrails AI, Ragas, DeepEval

**One-line summary:** Users ingest documents → system builds a vector-store-backed
knowledge base → a team of specialized AI agents (Researcher, Fact-Checker,
Synthesizer, Critic) orchestrated via LangGraph answers research questions with
evaluated, cited responses.

**Current Phase:** _{Phase 1 | Phase 2 | Phase 3 | Phase 4}_
**Phase Definition:** See `docs/PHASES.md` for full roadmap.

---

## 2. Documentation Index

<!-- AUDIENCE: ALL AGENTS — scan on first turn, deep-read only what's relevant -->

Before writing ANY code, check if relevant documentation already exists:

| # | Document | Purpose | When to Read |
|---|----------|---------|--------------|
| 1 | `docs/PRD.md` | Product requirements, user stories, acceptance criteria | Before building any user-facing feature |
| 2 | `docs/ARCHITECTURE.md` | System design, data flow diagrams, component relationships | Before creating new modules or changing data flow |
| 3 | `docs/PHASES.md` | Implementation roadmap, phase gates, definition of done | Before starting any phase or claiming a task |
| 4 | `docs/decisions.md` | Architecture Decision Records (ADRs) | Before making any architectural choice — check if it was already decided |
| 5 | `docs/PROGRESS.md` | Append-only session log with what was built, decided, and learned | At start of every session — read last 3 entries minimum |
| 6 | `docs/CONTRACTS.md` | Interface contracts between modules (types, signatures, protocols) | Before implementing any cross-module function call |
| 7 | `docs/flow.md` | Data flow from ingestion → retrieval → generation, file status tracking | Before modifying pipeline order or adding new pipeline stages |
| 8 | `AGENTS.md` | This file — rules, conventions, coordination protocol | Every session, every agent |

> **Rule:** If you need information that should be in one of these docs but isn't,
> **create it** — don't carry implicit knowledge that dies with your session.

---

## 3. Module Ownership Map

<!-- AUDIENCE: ALL AGENTS — read to know your boundaries -->

Each module has a designated **owner scope**. Agents working on a module MUST NOT
modify files outside their scope without explicit cross-reference in the session log.

```
src/
├── config/          # SCOPE: any agent (shared utilities)
│   ├── settings.py           → Pydantic Settings (env vars)
│   └── constants.py          → Enums, defaults, magic numbers
│
├── ingestion/       # SCOPE: ingestion-agent
│   ├── loaders.py            → Document loaders (PDF, web, MD, CSV)
│   ├── chunker.py            → Chunking strategies
│   ├── embedder.py           → Embedding pipeline
│   ├── deduplicator.py       → Content-hash dedup
│   └── pipeline.py           → Ingestion orchestrator
│
├── vectorstore/     # SCOPE: vectorstore-agent
│   ├── base.py               → Abstract VectorStore protocol
│   ├── chroma_store.py       → ChromaDB implementation
│   ├── pinecone_store.py     → Pinecone implementation
│   └── manager.py            → Factory + connection pooling
│
├── retrieval/       # SCOPE: retrieval-agent
│   ├── dense.py              → Dense vector retrieval
│   ├── sparse.py             → BM25 sparse retrieval
│   ├── hybrid.py             → RRF fusion
│   ├── reranker.py           → Cross-encoder re-ranking
│   ├── multi_query.py        → Query expansion
│   └── pipeline.py           → Retrieval orchestrator
│
├── agents/          # SCOPE: orchestration-agent
│   ├── state.py              → LangGraph state TypedDict
│   ├── graph.py              → LangGraph graph builder
│   ├── nodes.py              → Graph node implementations
│   ├── crew.py               → CrewAI agent definitions
│   ├── tools.py              → Agent tools
│   └── prompts.py            → System prompts
│
├── guardrails/      # SCOPE: quality-agent
│   ├── input_guards.py       → Input processing
│   ├── output_guards.py      → Output processing
│   └── validators.py         → Custom processors
│
├── evaluation/      # SCOPE: quality-agent
│   ├── datasets.py           → Golden dataset management
│   ├── ragas_eval.py         → Ragas runner
│   ├── deepeval_eval.py      → DeepEval runner
│   ├── reports.py            → Report generation
│   └── synthetic.py          → Synthetic data generation
│
├── finetuning/      # SCOPE: finetuning-agent
│   ├── curator.py            → Q&A pair curation
│   ├── formatter.py          → Dataset formatting
│   ├── trainer.py            → QLoRA training
│   ├── evaluator.py          → A/B comparison
│   └── exporter.py           → GGUF export
│
├── api/             # SCOPE: api-agent
│   ├── main.py               → FastAPI app
│   ├── routes/               → Endpoint handlers
│   └── middleware.py         → CORS, auth, rate limiting
│
└── ui/              # SCOPE: ui-agent
    ├── app.py                → Streamlit entry point
    ├── pages/                → Streamlit pages
    └── components/           → Reusable UI components
```

### Cross-Module Rules

- **Read from any module** — always allowed
- **Write to your own module** — always allowed
- **Write to `config/`** — allowed (shared, but append-only for constants)
- **Write to another agent's module** — PROHIBITED unless:
  1. You document the reason in `docs/PROGRESS.md`
  2. You only modify the **interface boundary** (function signatures in `__init__.py`)
  3. You update `docs/CONTRACTS.md` with the new contract

---

## 4. Interface Contracts

<!-- AUDIENCE: ALL AGENTS — read before calling any cross-module function -->

All cross-module communication MUST go through defined contracts. These are the
**stable API surfaces** between modules. Internal implementations can change
freely; these signatures MUST NOT change without updating this section AND
`docs/CONTRACTS.md`.

### 4.1 Ingestion → VectorStore

```python
# Contract: IngestionPipeline.ingest() returns chunks ready for storage
@dataclass
class ProcessedChunk:
    id: str                    # Deterministic hash of content + source
    text: str                  # Chunk text content
    embedding: list[float]     # Dense embedding vector
    metadata: ChunkMetadata    # Source, page, timestamp, etc.

@dataclass
class ChunkMetadata:
    source: str                # File path or URL
    source_type: str           # "pdf" | "web" | "markdown" | "csv"
    page: int | None           # Page number (PDFs only)
    section: str | None        # Section header if detected
    ingested_at: str           # ISO 8601 timestamp
    content_hash: str          # SHA-256 of raw text

# Ingestion output: list[ProcessedChunk]
# VectorStore input: list[ProcessedChunk]
```

### 4.2 VectorStore → Retrieval

```python
# Contract: VectorStore exposes a uniform search interface
class VectorStoreProtocol(Protocol):
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]: ...

    async def upsert(self, chunks: list[ProcessedChunk]) -> int: ...
    async def delete(self, ids: list[str]) -> int: ...

@dataclass
class SearchResult:
    id: str
    text: str
    metadata: ChunkMetadata
    score: float               # Similarity score (0-1, higher = better)
```

### 4.3 Retrieval → Agents

```python
# Contract: RetrievalPipeline returns ranked, re-ranked results
@dataclass
class RetrievalResult:
    query: str                          # Original query
    expanded_queries: list[str]         # Multi-query expansions
    results: list[SearchResult]         # Final ranked results (post re-rank)
    retrieval_metadata: RetrievalMeta   # Timing, strategy used

@dataclass
class RetrievalMeta:
    strategy: str              # "hybrid" | "dense" | "sparse"
    dense_results: int         # Count before fusion
    sparse_results: int        # Count before fusion
    reranked: bool             # Whether cross-encoder was applied
    latency_ms: float          # Total retrieval time
```

### 4.4 Agents → Guardrails

```python
# Contract: Guardrails wrap agent I/O
class GuardResult:
    passed: bool
    original_input: str
    validated_output: str | None    # None if rejected
    violations: list[Violation]
    action_taken: str               # "pass" | "fix" | "reject" | "reask"

@dataclass
class Violation:
    guard_name: str            # e.g. "Input Validation"
    severity: str              # "low" | "medium" | "high" | "critical"
    description: str           # Human-readable explanation
    span: tuple[int, int] | None  # Character offsets if applicable
```

### 4.5 Agents → Evaluation

```python
# Contract: Evaluation takes Q&A triples, returns scored metrics
@dataclass
class EvalSample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None   # None for production (no label)

@dataclass
class EvalResult:
    sample_id: str
    metrics: dict[str, float]  # {"faithfulness": 0.92, "relevancy": 0.88, ...}
    passed: bool               # All metrics above thresholds?
    timestamp: str
```

> **Adding a new contract?** Define it here first as pseudocode/types,
> THEN implement it. Never implement cross-module calls without a contract.

---

## 5. Verification Protocol

<!-- AUDIENCE: ALL AGENTS — MANDATORY, read every session -->

These rules exist because AI coding agents frequently misstate APIs, file
paths, package names, and function signatures. Every rule here was born from
a real failure mode.

### 5.1 The NEVER ASSUME Rule

```
BEFORE writing any code that:
  - Imports a module          → VERIFY the module exists (check file system)
  - Calls a function          → VERIFY the function signature (read the source)
  - Uses a library API        → VERIFY the API exists in the installed version
  - References a file path    → VERIFY the file exists (ls/find)
  - Uses a config key         → VERIFY it's defined in settings.py
  - Assumes a type/schema     → VERIFY against the contract in Section 4
```

### 5.2 Verification Commands

Run these BEFORE writing code, not after:

```bash
# Does this file exist?
ls -la src/path/to/file.py

# What's the actual function signature?
grep -n "def function_name" src/path/to/file.py

# What version of this library is installed?
pip show package-name | grep Version

# What does this library's API actually look like?
python -c "import module; help(module.ClassName)"

# What are the actual exports of this module?
grep -n "^def \|^class \|^async def " src/module/__init__.py
```

### 5.3 Mandatory Pre-Flight Checks

Before EVERY coding session, the agent MUST:

1. **Read `docs/PROGRESS.md`** (last 3 entries) — know what was done before you
2. **Read this file** (`AGENTS.md`) — rules may have changed
3. **Check `docs/CONTRACTS.md`** — interfaces may have been updated
4. **Run `uv run pytest --co -q`** — discover existing test structure
5. **Run `uv run python -c "import src; print('OK')"** — verify project imports

### 5.4 When You Don't Know

```
IF you are unsure about:
  - A library API         → Read the library docs or source, don't guess
  - A project convention  → Search existing code for patterns, don't invent
  - A design decision     → Check docs/decisions.md, don't re-decide
  - Whether code exists   → Search the codebase, don't duplicate

NEVER say "I believe this function exists" — VERIFY or CREATE it.
NEVER say "This should work" — TEST it.
NEVER assume a previous agent completed their work — CHECK the files.
```

### 5.5 Import Discipline

```python
# ✅ CORRECT — import from the package's public API
from src.vectorstore.manager import VectorStoreManager

# ❌ WRONG — importing from a file you haven't verified exists
from src.vectorstore.advanced_search import HybridSearchEngine  # DOES THIS EXIST?

# Rule: Before every import, mentally answer: "Have I seen this file? Yes/No."
# If No → check the filesystem first.
```

### 5.6 Dependency Management

```
BEFORE using any library:
  1. Check if it's in pyproject.toml
  2. If not, propose adding it with justification
  3. NEVER pip install directly — use `uv add package-name`
  4. Pin to a version range, not exact version
  5. After adding, run `uv sync` and verify import works
```

---

## 6. Coding Standards

<!-- AUDIENCE: ALL AGENTS — read once, internalize -->

### 6.1 Python Style

- **Type hints on ALL function signatures** — no exceptions
- **Pydantic models** for all data structures that cross module boundaries
- **`dataclass`** for internal-only data structures
- **`async/await`** for all I/O-bound operations (file reads, API calls, DB queries)
- **No `Any`** unless genuinely unavoidable — document why with a comment
- **Max function length:** 50 lines (excluding docstring). If longer, decompose.
- **Max file length:** 400 lines. If longer, split into sub-modules.

### 6.2 Naming Conventions

```python
# Files: snake_case.py
chroma_store.py       # ✅
ChromaStore.py        # ❌

# Classes: PascalCase
class VectorStoreManager:    # ✅
class vector_store_manager:  # ❌

# Functions: snake_case, verb-first
async def ingest_documents():    # ✅
async def documents_ingest():    # ❌

# Constants: SCREAMING_SNAKE_CASE
DEFAULT_CHUNK_SIZE = 512     # ✅
defaultChunkSize = 512       # ❌

# Private: single underscore prefix
def _compute_hash():         # ✅ internal helper
def __compute_hash():        # ❌ name mangling is rarely needed
```

### 6.3 Docstrings

```python
async def search(
    self,
    query: str,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
) -> list[SearchResult]:
    """Search the vector store with hybrid retrieval.

    Uses dense + sparse fusion with RRF. Results are re-ranked
    via cross-encoder if fewer than 20 candidates.

    Args:
        query: Natural language search query.
        top_k: Maximum number of results to return.
        filters: Optional metadata filters (e.g., {"source_type": "pdf"}).

    Returns:
        Ranked list of SearchResult, highest relevance first.

    Raises:
        VectorStoreConnectionError: If the store is unreachable.
        EmptyQueryError: If query is empty or whitespace-only.
    """
```

### 6.4 Error Handling

```python
# Define module-specific exceptions in each module's __init__.py
class IngestionError(Exception): ...
class ChunkingError(IngestionError): ...
class EmbeddingError(IngestionError): ...

# Use specific exceptions, never bare except
try:
    chunks = await chunker.split(document)
except ChunkingError as e:
    logger.error("Chunking failed", document=doc.source, error=str(e))
    raise
# ❌ except Exception: ...
# ❌ except: ...
```

### 6.5 Logging

```python
import structlog

logger = structlog.get_logger(__name__)

# Always use structured key-value logging
logger.info("Document ingested",
    source=doc.source,
    chunks=len(chunks),
    duration_ms=elapsed,
)
# ❌ logger.info(f"Ingested {doc.source} with {len(chunks)} chunks")
```

### 6.6 Comments

```python
# Comment WHY, not WHAT:

# ✅ Good — explains a non-obvious decision
# RRF k=60 gives best balance between dense and sparse results
# based on our eval runs (see docs/decisions.md#ADR-007)
k = 60

# ❌ Bad — restates the code
# Set k to 60
k = 60
```

---

## 7. Architecture Invariants

<!-- AUDIENCE: ALL AGENTS — NEVER violate without updating this section -->

These are hard rules. Violating them without updating this file is a bug.

1. **Vector store is always persistent.** No in-memory-only vector stores.
   ChromaDB must use `PersistentClient`. This is the whole point of the project.

2. **All LLM calls go through a unified provider abstraction.** Never call
   `openai.ChatCompletion.create()` directly. Use the `LLMProvider` protocol
   in `src/config/`. Switching providers must require only a config change.

3. **Embedding model is configurable, not hardcoded.** The embedding dimension
   is read from config, never assumed to be 1536.

4. **Agents never call the vector store directly.** They use the Retrieval
   module's public API. The vector store is an implementation detail of retrieval.

5. **Processing layers are middleware, not inline.** They wrap agent calls, they don't
   live inside agent logic. An agent should work identically with processing layers
   enabled or disabled.

6. **Evaluation never modifies production data.** Eval pipelines read from
   golden datasets and the production pipeline, but never write to the
   production vector store or modify production config.

7. **No hardcoded secrets.** All API keys, endpoints, and credentials come from
   environment variables via Pydantic Settings. `grep -r "sk-"` must return
   zero results.

8. **All async code uses `asyncio`.** No threading for I/O. No `concurrent.futures`
   for API calls. Threading is allowed only for CPU-bound operations (embedding
   computation on local models).

---

## 8. Commands

<!-- AUDIENCE: ALL AGENTS — reference as needed -->

```bash
# ─── Setup ───────────────────────────────────────────────────────────
uv sync                              # Install all dependencies
cp .env.example .env                  # Create env file (edit with keys)

# ─── Development ─────────────────────────────────────────────────────
uv run python -m src.api.main        # Start FastAPI server (dev)
uv run streamlit run src/ui/app.py   # Start Streamlit UI

# ─── Testing ─────────────────────────────────────────────────────────
uv run pytest                         # Run all tests
uv run pytest tests/unit/             # Unit tests only
uv run pytest tests/integration/      # Integration tests only
uv run pytest --co -q                 # List all tests (dry run)
uv run pytest -x -v                   # Stop on first failure, verbose

# ─── Quality ─────────────────────────────────────────────────────────
uv run ruff check src/                # Lint
uv run ruff format src/               # Format
uv run mypy src/                      # Type checking

# ─── Data ────────────────────────────────────────────────────────────
uv run python scripts/ingest_sample_data.py    # Load sample docs
uv run python scripts/run_evals.py             # Run evaluation suite
uv run python scripts/generate_synthetic.py    # Generate synthetic Q&A

# ─── Docker ──────────────────────────────────────────────────────────
docker compose up -d                  # Start all services
docker compose logs -f api            # Tail API logs
```

---

## 9. Testing Requirements

<!-- AUDIENCE: ALL AGENTS — read before writing tests or claiming done -->

### 9.1 Test Coverage Requirements

| Module | Minimum Coverage | What Must Be Tested |
|--------|-----------------|---------------------|
| `ingestion/` | 85% | Chunking logic, dedup, metadata extraction |
| `vectorstore/` | 80% | CRUD operations, search with filters, error cases |
| `retrieval/` | 90% | RRF fusion math, re-ranking order, empty results |
| `agents/` | 75% | State transitions, conditional edges, loop guards |
| `guardrails/` | 90% | Every processor with pass/fail inputs |
| `evaluation/` | 80% | Metric computation, report generation |
| `api/` | 80% | All endpoints, error responses, auth |

### 9.2 Test Structure

```python
# Tests mirror source structure
tests/
├── unit/                    # Fast, no I/O, no network
│   ├── test_chunker.py      # Test chunking with known inputs
│   ├── test_hybrid.py       # Test RRF math with fixed scores
│   └── test_guards.py       # Test each processor with edge cases
├── integration/             # May use local DB, mock APIs
│   ├── test_ingestion.py    # Full ingest pipeline with test docs
│   ├── test_retrieval.py    # End-to-end search with ChromaDB
│   └── test_agents.py       # Agent graph execution with mocked LLM
└── conftest.py              # Shared fixtures, test ChromaDB client
```

### 9.3 Test Naming Convention

```python
def test_{function_name}_{scenario}_{expected_result}():
    """Tests are named: what_when_then."""
    pass

# Examples:
def test_chunk_document_with_empty_input_returns_empty_list(): ...
def test_rrf_fusion_with_disjoint_results_merges_correctly(): ...
def test_input_guard_with_email_in_query_redacts_email(): ...
```

### 9.4 Definition of Done

A task is NOT done until:
- [ ] Code compiles: `uv run python -c "from src.module import X"`
- [ ] Tests pass: `uv run pytest tests/unit/test_relevant.py -v`
- [ ] Types check: `uv run mypy src/module/`
- [ ] Lint passes: `uv run ruff check src/module/`
- [ ] Contracts honored: cross-module calls match Section 4 signatures
- [ ] Progress logged: entry appended to `docs/PROGRESS.md`

---

## 10. Configuration

<!-- AUDIENCE: ALL AGENTS — mandatory rules -->

- **API keys** → Environment variables only (`.env` + Pydantic Settings).
  Never in source code, never in comments, never in test fixtures.
- **`.env`** → Listed in `.gitignore`. Never committed.
- **User documents** → Stored locally only. Never transmitted except to
  configured LLM/embedding endpoints.
- **Vector store credentials** → Environment variables. ChromaDB local needs
  no auth. Pinecone API key via `PINECONE_API_KEY`.
- **Pre-commit check:** `grep -rn "sk-\|api_key\s*=\s*[\"']" src/` must
  return zero results.

---

## 11. Session Memory & Progress

<!-- AUDIENCE: ALL AGENTS — update every session -->

### 11.1 Progress Log (`docs/PROGRESS.md`)

This is the **shared memory** between agents and sessions. It is **append-only**.

**Every agent MUST append an entry at the END of their session:**

```markdown
---

## Session: {YYYY-MM-DD HH:MM} — {Agent/Scope Identifier}

### What I Built
- Created `src/ingestion/chunker.py` with RecursiveCharacterTextSplitter
- Added `test_chunker.py` with 12 test cases (all passing)

### Decisions Made
- ADR-003: Chose 512-token chunks over 256 because eval showed +8% faithfulness
  (see `notebooks/01_rag_exploration.ipynb` cell 14)

### What Went Wrong
- Initially used `tiktoken` for token counting but it doesn't support the BGE
  tokenizer. Switched to `transformers.AutoTokenizer`. Lost ~30 min.

### Known Issues
- Semantic chunker is 3x slower than recursive. Need to benchmark whether
  the quality gain justifies it. Leaving it as opt-in for now.

### What's Next
- Implement `src/ingestion/embedder.py`
- Wire up chunker → embedder → vector store pipeline
- Need to decide: batch embed or stream embed? (see ADR-pending)

### Files Changed
- [NEW] `src/ingestion/chunker.py`
- [NEW] `tests/unit/test_chunker.py`
- [MOD] `src/config/constants.py` — added CHUNK_SIZE, CHUNK_OVERLAP
- [MOD] `docs/CONTRACTS.md` — added ProcessedChunk schema
```

### 11.2 Decision Records (`docs/decisions.md`)

**Any decision that would be non-obvious to a future agent MUST be recorded:**

```markdown
## ADR-{NNN}: {Decision Title}

**Date:** {YYYY-MM-DD}
**Status:** Accepted | Superseded by ADR-{NNN} | Deprecated
**Context:** {Why this decision was needed}
**Decision:** {What was decided}
**Consequences:** {Trade-offs, what this enables, what this prevents}
**Evidence:** {Benchmarks, test results, or reasoning}
```

### 11.3 State Snapshot

Agents should update this section at the end of their session to give the next
agent a quick status read:

```
CURRENT STATE (last updated: {timestamp})
─────────────────────────────────────────
Phase 1 — RAG Pipeline
  [x] Document loaders (PDF, Web, MD, CSV)
  [x] Chunking engine (recursive + semantic)
  [/] Embedding pipeline (OpenAI done, HF in progress)
  [ ] Vector store (ChromaDB)
  [ ] Retrieval engine
  [ ] Generation pipeline
  [ ] Streamlit UI

Phase 2 — Multi-Agent
  [ ] LangGraph state machine
  [ ] CrewAI agents
  [ ] Agent tools

Phase 3 — Processing & Evals
  [ ] Input processing
  [ ] Output processing
  [ ] Ragas eval pipeline
  [ ] DeepEval metrics
  [ ] Golden dataset

Phase 4 — Fine-Tuning
  [ ] Dataset curation
  [ ] QLoRA training
  [ ] A/B evaluation
  [ ] GGUF export
```

---

## 12. Multi-Agent Coordination Protocol

<!-- AUDIENCE: ALL AGENTS — critical for parallel work -->

### 12.1 Before Starting Work

```
1. Read docs/PROGRESS.md (last 3 entries)          — What happened?
2. Read the State Snapshot (Section 11.3)           — What's done?
3. Check docs/CONTRACTS.md                          — Did interfaces change?
4. Identify your scope (Section 3)                  — What can you touch?
5. Verify dependencies exist                        — Can you build on them?
   → If a module you depend on isn't built yet, BUILD IT or STOP.
     Do NOT write code against imaginary interfaces.
```

### 12.2 While Working

```
- Stay within your module scope (Section 3)
- If you need to change a contract → update docs/CONTRACTS.md FIRST
- If you discover a bug in another module → log it in docs/PROGRESS.md
  under "Known Issues", do NOT fix it (unless it blocks you)
- If you're blocked → document WHY in PROGRESS.md and stop
  Do NOT write speculative code around the blocker
```

### 12.3 After Finishing Work

```
1. Run all tests: uv run pytest -x -v
2. Run type check: uv run mypy src/your_module/
3. Run lint: uv run ruff check src/your_module/
4. Update State Snapshot (Section 11.3)
5. Append to docs/PROGRESS.md using the template in 11.1
6. If you made a non-obvious decision → add ADR to docs/decisions.md
7. If you changed a contract → verify docs/CONTRACTS.md is updated
```

### 12.4 Conflict Resolution

If two agents have made conflicting changes:

1. **Contracts win.** If `docs/CONTRACTS.md` defines a signature, that's the
   source of truth. The implementation must match the contract.
2. **Earlier ADR wins.** If a decision was already made (check `docs/decisions.md`),
   respect it unless you have a quantified reason to supersede it.
3. **Tests win.** If existing tests define behavior, don't change the behavior
   to match your implementation — fix your implementation.
4. **When in doubt, don't merge.** Document the conflict in `docs/PROGRESS.md`
   and let the next agent (or the human) resolve it.

---

## 13. Emergency Procedures

<!-- AUDIENCE: ALL AGENTS — reference when things go wrong -->

### Something is broken and I don't know why

```bash
# 1. Check if it ever worked
git log --oneline -10

# 2. Find what changed
git diff HEAD~1

# 3. Run the minimal reproduction
uv run pytest tests/unit/test_specific.py -x -v

# 4. Check dependencies
uv run pip check

# 5. Nuclear option — clean reinstall
rm -rf .venv
uv sync
```

### I accidentally modified a file outside my scope

```bash
# Revert only that file
git checkout HEAD -- src/other_module/file.py

# Document what happened in PROGRESS.md
```

### The vector store is corrupted

```bash
# ChromaDB: delete and re-ingest
rm -rf data/chroma_db/
uv run python scripts/ingest_sample_data.py
```

### Tests are failing but I didn't change anything

```bash
# Check if a dependency updated
uv lock --check

# Check if env vars are set
uv run python -c "from src.config.settings import settings; print(settings)"
```

---

## 14. Changelog

<!-- AUDIENCE: ALL AGENTS — append when you modify this file -->

Track all modifications to this AGENTS.md file:

| Date | Author/Agent | Change |
|------|-------------|--------|
| _{date}_ | _{you}_ | Initial creation |

---

> **Remember:** This file is a living document. If you discover a rule that's
> missing, a convention that's unclear, or a failure mode that should be
> documented — **add it here.** The next agent will thank you.

