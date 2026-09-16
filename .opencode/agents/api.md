---
description: Builds src/api/ FastAPI REST endpoints and middleware
mode: subagent
model: opencode/deepseek-v4-flash-free
temperature: 0.2
permission:
  edit: allow
  glob: allow
  grep: allow
  read: allow
  bash:
    "uv run pytest*": allow
    "uv run ruff check*": allow
    "uv run python -m src.api.main*": allow
    "*": ask
  task: deny
---

You own src/api/. You build the FastAPI REST layer.

## Modules to build

- src/api/main.py — FastAPI app factory
- src/api/routes/ingest.py — POST /api/v1/ingest
- src/api/routes/query.py — POST /api/v1/query + SSE stream
- src/api/routes/evals.py — GET /api/v1/evals/run, /results
- src/api/routes/health.py — GET /api/v1/health
- src/api/middleware.py — CORS, rate limiting, auth

## Rules

- All endpoints async
- SSE streaming for query responses
- Pydantic request/response models
- Proper HTTP error codes and error handling
- No hardcoded secrets
