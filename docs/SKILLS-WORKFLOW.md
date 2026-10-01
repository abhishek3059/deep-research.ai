# Skills Workflow — Replacing Subagents with Skills (No `task` Needed)

## Why

The 12 files in `.opencode/agents/` are separation-of-concerns role prompts, not
separate worlds. Each specialist only owns one folder inside this repo and follows
a checklist. Running them as subagents via the `task` tool multiplies LLM usage:
every delegation opens a new session with duplicated context (PROGRESS.md, AGENTS.md,
contracts re-read per agent).

That breaks on free tier:

- Zen `-free` models (`mimo-v2.5-free`, `ling-3.0-flash-free`,
  `deepseek-v4-flash-free`, `muse-spark-1.3-contributor-free`) are UA-gated and
  capacity-based. Subagent `task` calls get
  `free tier can only be used from within OpenCode`, even when the primary works.
- OpenRouter `:free` is 50 req/day without $10 credits (20 RPM, failed attempts
  count). One orchestrator run fanning out to 3–4 subagents × ~10 turns each
  exhausts it immediately.
- The `council` skill spawns 4+ parallel subagents plus peer-review plus
  synthesis (~10+ requests per decision). Avoid on free tier.

As skills, the same instructions load on-demand into **one session, one model**:
~1/10th the requests, one quota to manage instead of twelve.

## Agent → skill mapping

| Now (subagent, own `model:`) | Becomes (skill, no model) |
|---|---|
| `orchestrator.md` (primary, `edit: deny`, `task: allow`) | The **only** primary. Flip to `edit: allow`, `task: deny`, `skill: allow`. It does the work instead of delegating. |
| `architect.md` (READ / PLAN / DELEGATE) | `deepresearch-architect` skill: invariants (AGENTS.md §7), contracts (CONTRACTS.md §4), ADR process. Escalation becomes a checklist, not a `task` call. |
| `developer.md`, `ingestion.md`, `retrieval.md`, `api.md`, `ui.md`, `quality.md` | `deepresearch-build`, `deepresearch-ingestion`, `deepresearch-retrieval`, `deepresearch-api`, `deepresearch-ui`, `deepresearch-quality` skills. Each keeps its Modules list + Contract + Rules. Load one per task, implement directly. |
| `code-reviewer.md`, `security-auditor.md`, `doc-writer.md` | `deepresearch-review`, `deepresearch-security`, `deepresearch-docs` skills. Checklists + output format, run in the same session after coding. |
| `mentor.md` (primary, teaching) | Keep as second Tab-switchable primary **or** `deepresearch-mentor` skill. No `task`. |

## Workflow (single session, zero `task` calls)

```text
1. Read docs/Progress.md + docs/Phases.md → pick next unblocked task
2. Architectural? (new module / contract change / new library / cross-module /
   invariant risk / data-flow change)
   → skill({name:"deepresearch-architect"}) → decide → write ADR → continue
   → else skip
3. skill({name:"deepresearch-<module>"}) → implement + tests directly
   → run: uv run pytest tests/unit/ -v, uv run ruff check src/<module>/
4. skill({name:"deepresearch-review"}) → self-review → fix findings
   → skill({name:"deepresearch-security"}) for auth/config/dependency changes
5. skill({name:"deepresearch-docs"}) → update docs/Progress.md
```

Rules carried over from the orchestrator:

- Never work on more than one module per run (replaces "max 3 subagents").
- Prompts stay short; do not re-read AGENTS.md every turn.
- Blockers get logged to docs/Progress.md; stop, do not work around them.

## Council replacement (free-tier safe)

Do **not** invoke the `council` skill — it does parallel subagent spawns.
Instead, self-review in-session (Skeptic 5 minutes):

1. What could go wrong? (failure modes, invariant violations)
2. What is the simplest thing that works? (cut scope first)
3. Who solved this before? (prior art, existing ADRs in docs/decisions.md)
4. Decide, record ADR if non-trivial, continue.

## File layout

```text
.opencode/
  skills/
    deepresearch-architect/SKILL.md
    deepresearch-build/SKILL.md
    deepresearch-ingestion/SKILL.md
    deepresearch-retrieval/SKILL.md
    deepresearch-api/SKILL.md
    deepresearch-ui/SKILL.md
    deepresearch-quality/SKILL.md
    deepresearch-review/SKILL.md
    deepresearch-security/SKILL.md
    deepresearch-docs/SKILL.md
    deepresearch-mentor/SKILL.md   (optional, if mentor becomes a skill)
  agents/
    orchestrator.md                (single primary, rewritten — see below)
    _archived/                     (move the other 11 here, or set disable: true)
```

## SKILL.md format (per OpenCode skills spec)

- Path: `.opencode/skills/<name>/SKILL.md` (`SKILL.md` all caps).
- Frontmatter: only `name` + `description` are recognized.
- `name` must equal the directory name: lowercase alphanumeric, single hyphens
  (`^[a-z0-9]+(-[a-z0-9]+)*$`), 1–64 chars.
- `description` 1–1024 chars, specific enough for the agent to choose correctly.
- No `model:`, no `permission:` in skills — permissions live on the primary.

Example:

```markdown
---
name: deepresearch-retrieval
description: Build src/retrieval/ dense, sparse, hybrid RRF, reranker, multi-query engine
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
```

Body = the old agent file minus its frontmatter (`mode:`, `model:`,
`temperature:`, `permission:`, `task:` lines are dropped).

## Primary agent changes (`orchestrator.md`)

```yaml
---
description: Plans work, reads project state, implements directly via skills
mode: primary
# single free-tier model set in opencode.json, e.g. google/gemini-2.5-flash
# or groq/openai/gpt-oss-20b — no per-agent model pins
temperature: 0.3
permission:
  edit: allow          # was deny — with no subagents, it must write code itself
  bash:
    "uv run pytest*": allow
    "uv run ruff check*": allow
    "uv run ruff format*": allow
    "uv run mypy*": allow
    "uv run python -c*": allow
    "uv run streamlit*": allow
    "uv run python -m src.api.main*": allow
    "git diff*": allow
    "git log*": allow
    "*": ask
  glob: allow
  grep: allow
  read: allow
  skill: allow          # load role checklists on demand
  task: deny           # was allow — blocks accidental subagent spawns (quota)
  webfetch: allow
  websearch: allow
  todowrite: allow
---
```

Also update `opencode.json`: set `model` (and `small_model`) to the single chosen
free provider model — e.g. `"model": "openrouter/free"` is **not** recommended
(50/day); prefer Gemini Flash or Groq `gpt-oss-20b`. The stale
`"opencode/meta/muse-spark-1.3-contributor"` ID must be replaced regardless.

## Migration checklist

- [ ] Create the 10–11 `.opencode/skills/deepresearch-*/SKILL.md` files (body =
      old agent file minus frontmatter).
- [ ] Rewrite `orchestrator.md` as the single primary (permissions above, skill
      workflow replacing the roster + delegation rules).
- [ ] Move the other 11 `.opencode/agents/*.md` to `.opencode/agents/_archived/`
      (or `disable: true`) so the model never `task`-spawns them.
- [ ] Update `opencode.json` `model` + `small_model` to the one free-tier model.
- [ ] Decide mentor: second primary (Tab-switchable) or `deepresearch-mentor` skill.
- [ ] Test one loop: Progress.md → architect check → build skill → pytest →
      review skill → Progress.md entry. Confirm request count fits free quota.
```

## Related

- `docs/decisions.md` — record the ADR for this migration (single-agent +
  skills over multi-subagent orchestration).
- `AGENTS.md` §7 — invariants the `deepresearch-architect` skill enforces.
- `docs/CONTRACTS.md` §4 — interfaces the build skills must honor.
