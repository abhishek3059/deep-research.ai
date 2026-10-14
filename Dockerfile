# DeepResearch AI — single-container deployment (API :8000 + UI :8501).
# Python 3.13 is pinned in .python-version; the slim image matches it.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install uv, then dependencies first for layer caching. Production deps only
# (dev extra is CI-local; tests never run inside the image).
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy the application (data/ is a mounted volume on Render, not baked in).
COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/
COPY docs/ docs/
COPY README.md ./

EXPOSE 8000 8501

# API in the background, Streamlit in the foreground (keeps the container alive).
CMD ["sh", "-c", "uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & uv run streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0"]
