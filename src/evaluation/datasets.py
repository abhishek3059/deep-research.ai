"""Golden dataset management for evaluation.

Read-only with respect to production data: this module only reads from and
writes to the configured dataset directory (default ``data/golden_datasets``).
It never touches the production vector store or production config
(AGENTS.md invariant #6).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_DATASET_DIR = Path("data/golden_datasets")
STARTER_DATASET_NAME = "starter_dataset.json"


@dataclass
class EvalSample:
    """A single Q&A evaluation triple.

    Matches docs/CONTRACTS.md section 4.5.
    """

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None


@dataclass
class EvalResult:
    """Scored metrics for a single evaluated sample.

    Matches docs/CONTRACTS.md section 4.5.
    """

    sample_id: str
    metrics: dict[str, float]
    passed: bool
    timestamp: str = field(default_factory=lambda: _utc_now())


def _utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def sample_id_for(question: str, index: int = 0) -> str:
    """Derive a stable sample id from the question text and index.

    Args:
        question: The sample question.
        index: Position in the dataset (disambiguates duplicate questions).

    Returns:
        Short deterministic id like ``sample-3f9a1c2e``.
    """
    digest = hashlib.sha256(f"{index}:{question}".encode()).hexdigest()[:8]
    return f"sample-{digest}"


def _validate_record(record: dict[str, object], index: int) -> EvalSample:
    """Validate a raw JSON record and convert it to an EvalSample.

    Args:
        record: Decoded JSON object for one dataset row.
        index: Row position, used in error messages.

    Returns:
        Validated EvalSample.

    Raises:
        ValueError: If required fields are missing or mistyped.
    """
    if not isinstance(record, dict):
        raise ValueError(f"Row {index}: expected a JSON object")
    question = record.get("question")
    answer = record.get("answer")
    contexts = record.get("contexts")
    ground_truth = record.get("ground_truth")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Row {index}: 'question' must be a non-empty string")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"Row {index}: 'answer' must be a non-empty string")
    if not isinstance(contexts, list) or not all(isinstance(c, str) for c in contexts):
        raise ValueError(f"Row {index}: 'contexts' must be a list of strings")
    if ground_truth is not None and not isinstance(ground_truth, str):
        raise ValueError(f"Row {index}: 'ground_truth' must be a string or null")
    return EvalSample(
        question=question,
        answer=answer,
        contexts=list(contexts),
        ground_truth=ground_truth,
    )


class GoldenDatasetManager:
    """Load, save, and curate golden Q&A datasets from JSON files."""

    def __init__(self, dataset_dir: Path | str = DEFAULT_DATASET_DIR) -> None:
        """Bind the manager to a dataset directory (created on demand).

        Args:
            dataset_dir: Directory holding ``*.json`` golden datasets.
        """
        self._dataset_dir = Path(dataset_dir)

    @property
    def dataset_dir(self) -> Path:
        """Return the directory this manager reads/writes."""
        return self._dataset_dir

    def list_datasets(self) -> list[str]:
        """List available ``*.json`` dataset file names."""
        if not self._dataset_dir.exists():
            return []
        return sorted(p.name for p in self._dataset_dir.glob("*.json") if p.is_file())

    def load(self, name: str) -> list[EvalSample]:
        """Load and validate a golden dataset by file name.

        Args:
            name: JSON file name inside the dataset directory.

        Returns:
            Validated list of EvalSample objects.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
            ValueError: If the JSON is malformed or rows fail validation.
        """
        path = self._dataset_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Golden dataset not found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}") from e
        if not isinstance(raw, list):
            raise ValueError(f"Dataset {path} must be a JSON array of records")
        samples = [_validate_record(record, i) for i, record in enumerate(raw)]
        logger.info("Golden dataset loaded", name=name, samples=len(samples))
        return samples

    def save(self, samples: list[EvalSample], name: str) -> Path:
        """Persist samples to a JSON file inside the dataset directory.

        Only ever writes inside the dataset directory — production data
        is never touched.

        Args:
            samples: Samples to persist.
            name: Target JSON file name.

        Returns:
            Path of the written file.
        """
        self._dataset_dir.mkdir(parents=True, exist_ok=True)
        path = self._dataset_dir / name
        payload = [asdict(s) for s in samples]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Golden dataset saved", name=name, samples=len(samples))
        return path

    def add_sample(self, sample: EvalSample, dataset_name: str) -> Path:
        """Append one curated sample to a dataset file.

        Creates the file if it does not exist yet.

        Args:
            sample: The new golden sample.
            dataset_name: Target JSON file name.

        Returns:
            Path of the updated file.
        """
        try:
            samples = self.load(dataset_name)
        except FileNotFoundError:
            samples = []
        samples.append(sample)
        return self.save(samples, dataset_name)

    @classmethod
    def create_starter_dataset(cls, dataset_dir: Path | str = DEFAULT_DATASET_DIR) -> Path:
        """Write the built-in starter dataset (12 research Q&A pairs).

        Args:
            dataset_dir: Directory to write the starter file into.

        Returns:
            Path of the written starter dataset.
        """
        manager = cls(dataset_dir)
        return manager.save(_starter_samples(), STARTER_DATASET_NAME)


def _starter_samples() -> list[EvalSample]:
    """Return the built-in starter Q&A pairs (factual, multi-hop, edge)."""
    return [
        EvalSample(
            question="What does RRF stand for in hybrid retrieval?",
            answer="RRF stands for Reciprocal Rank Fusion, a method that merges "
            "ranked lists from dense and sparse retrieval using 1/(k + rank).",
            contexts=[
                "Reciprocal Rank Fusion (RRF) merges ranked lists from dense and "
                "sparse retrieval by scoring each document as 1/(k + rank)."
            ],
            ground_truth="Reciprocal Rank Fusion",
        ),
        EvalSample(
            question="Which vector store backend does DeepResearch AI use for local persistence?",
            answer="DeepResearch AI uses ChromaDB with a PersistentClient so the "
            "knowledge base survives restarts.",
            contexts=[
                "The vector store is always persistent. ChromaDB must use "
                "PersistentClient so the knowledge base survives restarts."
            ],
            ground_truth="ChromaDB with PersistentClient",
        ),
        EvalSample(
            question="What is the default chunk size in the ingestion pipeline?",
            answer="The default chunk size is 512 tokens with a 64-token overlap.",
            contexts=["Chunking uses 512-token chunks with a 64-token overlap by default."],
            ground_truth="512 tokens with 64-token overlap",
        ),
        EvalSample(
            question="What are the Ragas quality targets for faithfulness and relevancy?",
            answer="The Ragas targets are faithfulness above 0.85 and answer relevancy above 0.90.",
            contexts=["Ragas targets: faithfulness > 0.85, relevancy > 0.90."],
            ground_truth="faithfulness > 0.85, relevancy > 0.90",
        ),
        EvalSample(
            question="What is BM25 used for in the retrieval pipeline?",
            answer="BM25 is the sparse retrieval method that complements dense "
            "vector search before RRF fusion.",
            contexts=[
                "Sparse retrieval uses BM25 keyword matching to complement dense "
                "vector search; both lists are fused with RRF."
            ],
            ground_truth="BM25 sparse keyword retrieval",
        ),
        EvalSample(
            question="How do chunking strategy and RRF fusion together affect answer faithfulness?",
            answer="Semantic chunking keeps related facts in one chunk, and RRF "
            "fusion surfaces chunks ranked highly by both dense and sparse "
            "retrieval, so the generator receives grounded context and "
            "faithfulness improves.",
            contexts=[
                "Semantic chunking keeps related facts within a single chunk.",
                "RRF fusion surfaces documents ranked highly by both dense and "
                "sparse retrieval for grounded generation.",
            ],
            ground_truth="Better grounded context raises faithfulness",
        ),
        EvalSample(
            question="Why does evaluation need to run before multi-agent orchestration?",
            answer="ADR-003 requires proving retrieval quality with a Ragas "
            "baseline first, so multi-agent complexity is only added when "
            "measurements justify it.",
            contexts=[
                "ADR-003 'Prove Before You Build' requires a Ragas baseline on "
                "Phase 1 before starting Phase 2 multi-agent work.",
                "Multi-agent complexity must be earned through evidence, not assumed.",
            ],
            ground_truth="ADR-003 evaluation-first decision",
        ),
        EvalSample(
            question="Which pipeline stages run between a user query and a cited answer?",
            answer="The query is expanded, embedded for dense retrieval, matched "
            "with BM25 sparse retrieval, fused with RRF, reranked with a "
            "cross-encoder, then passed with context to the generation pipeline.",
            contexts=[
                "Retrieval workflow: expand, dense plus sparse retrieval, RRF "
                "fusion, cross-encoder rerank, then generation with cited sources.",
            ],
            ground_truth="expand, retrieve, fuse, rerank, generate",
        ),
        EvalSample(
            question="How does the generation pipeline produce cited answers?",
            answer="It formats retrieved chunks as numbered sources, builds "
            "grounded prompts, calls the LLM provider, and returns the answer "
            "with source references.",
            contexts=[
                "Generation formats retrieved chunks into numbered sources and "
                "builds grounded prompts before calling the LLM provider.",
            ],
            ground_truth="Numbered sources plus grounded prompts",
        ),
        EvalSample(
            question="What is it?",
            answer="I cannot answer this precisely because the question is "
            "ambiguous. Please specify which component you mean.",
            contexts=[],
            ground_truth=None,
        ),
        EvalSample(
            question="What is the capital of France?",
            answer="Paris is the capital of France, but this is outside the "
            "research knowledge base, so the answer is not grounded in "
            "retrieved context.",
            contexts=[
                "The knowledge base covers RAG pipelines, vector stores, and "
                "evaluation metrics for DeepResearch AI."
            ],
            ground_truth=None,
        ),
        EvalSample(
            question="Summarize everything ever written about science in one sentence.",
            answer="That request is too broad for the knowledge base. A useful "
            "answer needs a narrower question scoped to ingested documents.",
            contexts=[],
            ground_truth=None,
        ),
    ]
