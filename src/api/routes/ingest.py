"""Document ingestion endpoint."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.ingestion.pipeline import IngestionPipeline

router = APIRouter()

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


def _get_pipeline() -> IngestionPipeline:
    return IngestionPipeline()


@router.post("/api/v1/ingest")
async def ingest_document(file: UploadFile = File(...)) -> dict[str, int]:  # noqa: B008
    """Ingest a single document and return chunk count.

    Accepts PDF, markdown, CSV, or plain text files.
    """
    pipeline = _get_pipeline()
    suffix = Path(file.filename or "upload.txt").suffix or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum upload size of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB",
            )
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks = await pipeline.ingest(tmp_path)
        return {"chunk_count": len(chunks)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)