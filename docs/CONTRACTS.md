# Interface Contracts

> All cross-module communication MUST go through these defined contracts.
> These are the **stable API surfaces** between modules. Internal
> implementations can change freely; these signatures MUST NOT change
> without updating this document.

---

## 4.1 Ingestion -> VectorStore

```python
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
```

---

## 4.2 VectorStore -> Retrieval

```python
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

---

## 4.3 Retrieval -> Agents

```python
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

---

## 4.4 Agents -> Guardrails

```python
@dataclass
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

---

## 4.5 Agents -> Evaluation

```python
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

---

> Adding a new contract? Define it here first as pseudocode/types,
> THEN implement it. Never implement cross-module calls without a contract.
