# DeepResearch AI

**Multi-Agent Research Platform with Persistent Knowledge Base**

A production-grade system that ingests user documents into a persistent vector store, orchestrates specialized AI agents via LangGraph, and delivers evaluated, cited research responses.

## Key Features

- Document ingestion (PDF, Web, Markdown, CSV)
- Persistent vector store (ChromaDB)
- Hybrid retrieval (dense + sparse with RRF fusion)
- LangGraph orchestration with self-critique loop
- FastAPI REST API
- Streamlit chat UI
- Ragas + DeepEval evaluation metrics

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/deep-research.ai.git
cd deep-research.ai

# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run the app
uv run python -m src.api.main
```

## Tech Stack

- Python 3.11+
- LangChain / LangGraph
- ChromaDB
- FastAPI
- Streamlit
- Ragas / DeepEval

## License

MIT
