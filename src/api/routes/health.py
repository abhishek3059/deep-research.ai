"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/api/v1/health")
async def health_check() -> dict[str, str]:
    """Return service health status and version."""
    return {"status": "ok", "version": "0.1.0"}
