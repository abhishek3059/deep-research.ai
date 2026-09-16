"""FastAPI application entry point for DeepResearch AI."""

from __future__ import annotations

from fastapi import FastAPI

from src.api.middleware import setup_middleware
from src.api.routes import health_router, ingest_router, query_router

app = FastAPI(
    title="DeepResearch AI",
    description="Multi-agent research platform with persistent knowledge base",
    version="0.1.0",
)

# Add middleware
setup_middleware(app)

# Include routers
app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(query_router)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint returning service name."""
    return {"service": "DeepResearch AI"}
