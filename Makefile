.PHONY: setup dev api ui demo test lint typecheck eval deploy

setup:
	uv sync --extra dev
	cp .env.example .env

api:
	uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000

ui:
	uv run streamlit run src/ui/app.py

demo:
	uv run python scripts/demo.py

test:
	uv run pytest -x -v

lint:
	uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/

typecheck:
	uv run mypy src/

eval:
	uv run python scripts/run_evals.py

deploy:
	docker build -t deepresearch-ai .
	docker run -p 8000:8000 -p 8501:8501 --env-file .env -v ./data:/app/data deepresearch-ai
