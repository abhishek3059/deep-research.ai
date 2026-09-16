# DeepResearch AI — A Novice's Guide to LangChain, RAG, and AI Retrieval

> **Audience:** Python developers new to LangChain, RAG, and AI retrieval systems.
> **Prerequisites:** Basic Python knowledge (functions, classes, async/await).
> **Project:** DeepResearch AI — a multi-agent research platform with a persistent knowledge base.

---

## Table of Contents

1. [What is RAG (Retrieval-Augmented Generation)?](#1-what-is-rag-retrieval-augmented-generation)
2. [Project Overview — DeepResearch AI](#2-project-overview--deepresearch-ai)
3. [The Document Ingestion Pipeline](#3-the-document-ingestion-pipeline)
4. [Vector Stores — The Memory of RAG](#4-vector-stores--the-memory-of-rag)
5. [The Retrieval Engine — Finding Relevant Information](#5-the-retrieval-engine--finding-relevant-information)
6. [Generation — Turning Retrieved Context into Answers](#6-generation--turning-retrieved-context-into-answers)
7. [The API Layer](#7-the-api-layer)
8. [The Streamlit UI](#8-the-streamlit-ui)
9. [Configuration & Settings](#9-configuration--settings)
10. [Key Concepts Glossary](#10-key-concepts-glossary)
11. [Architecture Diagram](#11-architecture-diagram)
12. [What's Next (Phase 2-4 Preview)](#12-whats-next-phase-24-preview)

---

## 1. What is RAG (Retrieval-Augmented Generation)?

### The Problem

Large Language Models like GPT-4 are incredibly powerful, but they have two critical limitations:

1. **Knowledge cutoffs** — They only know about information published before their training data ends. Ask about a document you wrote last week, and they'll have no idea.
2. **No access to your private data** — Your company's internal docs, your research papers, your personal notes — the LLM has never seen any of it.

If you ask "What does our Q3 revenue report say about European markets?" to a bare LLM, it will either hallucinate an answer or tell you it doesn't know.

### The Solution: RAG

**Retrieval-Augmented Generation (RAG)** solves this by combining two steps:

1. **Retrieve** — First, search your documents for chunks that are relevant to the question.
2. **Generate** — Then, feed those chunks as context to the LLM and ask it to answer based *only* on that evidence.

> **Analogy: The Librarian**
>
> Imagine you walk into a library and ask a question. A regular LLM is like a librarian who has memorized every book published before 2024 but has never visited *this* library. A RAG system is like a librarian who *first walks to the shelves*, finds the 5 most relevant books, opens them to the right pages, and *then* answers your question while pointing at the exact passages.

### RAG vs. Fine-Tuning

| | RAG | Fine-Tuning |
|---|---|---|
| **How it works** | Retrieve documents at query time, include in prompt | Retrain the model on your data |
| **Data freshness** | Always up-to-date (just re-ingest) | Stale until you retrain |
| **Cost** | Low (no GPU training) | High (GPU hours) |
| **Transparency** | Cites sources — you can verify | Black box — hard to trace |
| **Best for** | Document Q&A, research | Style adaptation, domain jargon |

DeepResearch AI uses RAG for Phase 1, and plans fine-tuning in Phase 4.

> **Key Takeaway:** RAG = "find relevant documents first, then answer." It keeps the LLM grounded in real evidence and makes hallucination much harder.

---

## 2. Project Overview — DeepResearch AI

### What the Project Does

DeepResearch AI is a multi-agent research platform. Here's the user journey:

```
User uploads documents (PDFs, web pages, markdown, CSV)
        ↓
System ingests them into a persistent vector knowledge base
        ↓
User asks a research question
        ↓
AI agents retrieve relevant chunks, verify facts, synthesize an answer
        ↓
User receives a cited, grounded response
```

### The Four Phases

```
Phase 1 — Foundation RAG Pipeline        ███████████ DONE (41 tests passing)
Phase 2 — Multi-Agent Orchestration      ░░░░░░░░░░░ Coming next
Phase 3 — Safety Layer & Evaluation      ░░░░░░░░░░░ Planned
Phase 4 — Fine-Tuning & Self-Improvement ░░░░░░░░░░░ Planned
```

### Current State (Phase 1 Complete)

Phase 1 delivers a fully working RAG pipeline:

- **4 ingestion loaders** (PDF, text, CSV, markdown, web)
- **2 chunking strategies** (recursive + semantic)
- **Persistent ChromaDB** vector store with HNSW indexing
- **Hybrid retrieval** (dense + sparse with RRF fusion + cross-encoder reranking)
- **Generation pipeline** with OpenAI/Anthropic support and source citations
- **FastAPI REST API** with `/health`, `/ingest`, `/query` endpoints
- **Streamlit UI** with chat interface and file upload
- **41 passing tests**, lint-clean codebase

> **Key Takeaway:** DeepResearch AI takes your documents, builds a searchable knowledge base, and lets AI agents answer questions with cited evidence. Phase 1 (the foundation) is complete.

---

## 3. The Document Ingestion Pipeline

### Why Ingestion Matters

Before the system can answer questions, it needs to *read and understand* your documents. Ingestion is the process of turning raw files into searchable chunks stored in a vector database.

> **Analogy: Building a Library**
>
> Ingestion is like a library processing new acquisitions. The librarian doesn't just throw books on a random shelf — they:
> 1. **Open** each package (Load)
> 2. **Catalog** individual chapters and sections (Chunk)
> 3. **Remove duplicates** if two copies arrive (Deduplicate)
> 4. **Create index cards** describing what each section is about (Embed)

### The 4-Stage Pipeline

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│   LOAD   │ →  │  CHUNK   │ →  │  DEDUP   │ →  │  EMBED   │
│          │    │          │    │          │    │          │
│ Read raw │    │ Split    │    │ Remove   │    │ Convert  │
│ files    │    │ into     │    │ duplicate│    │ text to  │
│          │    │ pieces   │    │ chunks   │    │ vectors  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### 3a. Loaders (`src/ingestion/loaders.py`)

The `DocumentLoader` class reads raw files and produces LangChain `Document` objects. Each file type has its own loader because the format parsing is fundamentally different:

| Format | Loader | Why Different |
|--------|--------|---------------|
| **PDF** | `PyPDFLoader` | Extracts text per page, handles encoding |
| **Text** | `TextLoader` | Reads plain text with encoding detection |
| **CSV** | `CSVLoader` | One document per row, handles delimiters |
| **Markdown** | `UnstructuredMarkdownLoader` | Parses headings, code blocks, structure |
| **Web** | `WebBaseLoader` | Fetches HTML, extracts readable content |

Here's how the project detects and loads each type:

```python
# From src/ingestion/loaders.py
_EXTENSION_TO_SOURCE_TYPE: dict[str, SourceType] = {
    ".pdf": SourceType.PDF,
    ".txt": SourceType.TEXT,
    ".csv": SourceType.CSV,
    ".md": SourceType.MARKDOWN,
}

async def load(self, source: str, source_type: SourceType | None = None) -> list[Document]:
    """Load a single source, dispatching on its detected type."""
    if source_type is None:
        source_type = detect_source_type(source)
    if source_type is SourceType.PDF:
        return await self.load_pdf(source)
    if source_type is SourceType.TEXT:
        return await self.load_text(source)
    # ... etc.
```

> **Note:** All loaders run via `asyncio.to_thread()` because LangChain's file loaders are synchronous. This keeps the event loop free for other async work.

### 3b. Chunking (`src/ingestion/chunker.py`)

LLMs have limited context windows. You can't stuff an entire 200-page PDF into a prompt. **Chunking** splits documents into smaller, overlapping pieces that fit in the context window.

**Two strategies are available:**

1. **Recursive Character Splitting** (default, fast) — Splits on paragraph boundaries, then sentences, then characters. Configured with `chunk_size=512` characters and `chunk_overlap=64` characters.

2. **Semantic Splitting** (slower, smarter) — Uses an embedding model to find where the *meaning* changes between sentences.

**Why overlap matters:**

```
Without overlap (information lost at boundaries):
┌──────────────────┐┌──────────────────┐
│ ...end of chunk 1││start of chunk 2...│
│      "The policy"││"covers dental."  │
└──────────────────┘└──────────────────┘
  → The sentence "The policy covers dental." is split across two chunks.
     Neither chunk contains the full idea.

With overlap=64 (context preserved):
┌──────────────────────────┐
│ ...end of chunk 1        │
│      "The policy covers  │
│         dental."         │
│    OVERLAP →"dental. The │
│         new policy..."   │
└──────────────────────────┘
  → Adjacent chunks share 64 characters, so the full sentence appears
     in at least one chunk.
```

```python
# From src/ingestion/chunker.py
class TextChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
```

> **Key Takeaway:** 512-character chunks with 64-character overlap is the default. Recursive splitting is fast and good enough for most cases. Use semantic splitting when document structure matters.

### 3c. Deduplication (`src/ingestion/deduplicator.py`)

When you ingest the same document twice (or overlapping pages from a PDF), you get duplicate chunks. Deduplication uses **SHA-256 content hashing** to remove exact duplicates:

```python
# From src/ingestion/deduplicator.py
class Deduplicator:
    @staticmethod
    def compute_hash(text: str) -> str:
        """Return the hex-encoded SHA-256 digest of text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def deduplicate(self, documents: list[Document]) -> list[Document]:
        """Return documents with duplicate content removed."""
        seen: set[str] = set()
        unique: list[Document] = []
        for document in documents:
            digest = self.compute_hash(document.page_content)
            if digest in seen:
                continue
            seen.add(digest)
            unique.append(document)
        return unique
```

**Why dedup runs after chunking:** Chunks are the unit you store. Two full documents might both contain the phrase "Executive Summary" — but after chunking, those become distinct chunks with different surrounding context. Deduplicating at the chunk level is more precise.

### 3d. Embedding (`src/ingestion/embedder.py`)

**Embeddings** are the magic that makes RAG possible. An embedding model converts text into a list of numbers (a **vector**) that captures the *meaning* of the text.

> **Analogy: GPS Coordinates for Meaning**
>
> Think of an embedding as a GPS coordinate, but for meaning instead of location. "The cat sat on the mat" and "A feline rested on the rug" would get similar coordinates because they mean similar things. "Quantum computing breakthrough" would get a very different coordinate.

The project uses OpenAI's `text-embedding-3-small` model by default:

```python
# From src/ingestion/embedder.py
class Embedder:
    def __init__(self, model: str | None = None, batch_size: int = 100) -> None:
        self._batch_size = batch_size
        self._embeddings = embeddings or OpenAIEmbeddings(model=resolved_model)

    async def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        """Embed documents in input order."""
        return await self.embed_texts([doc.page_content for doc in documents])

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        return await self._embeddings.aembed_query(query)
```

**Batch processing** sends 100 texts at a time to the API to avoid rate limits and reduce overhead.

### 3e. Pipeline Orchestrator (`src/ingestion/pipeline.py`)

The `IngestionPipeline` wires all four stages together:

```python
# From src/ingestion/pipeline.py
class IngestionPipeline:
    async def ingest(self, source: str) -> list[ProcessedChunk]:
        """Run the full pipeline: load → chunk → dedup → embed."""
        documents = await self._loader.load(source)           # 1. Load
        chunks = await self._chunker.achunk_recursive(documents)  # 2. Chunk
        unique_chunks = self._deduplicator.deduplicate(chunks)    # 3. Dedup
        embeddings = await self._embedder.embed_documents(unique_chunks)  # 4. Embed
        return self._build_processed_chunks(unique_chunks, embeddings, source)
```

The output is a list of `ProcessedChunk` objects, each containing:
- `id` — deterministic hash (content + source)
- `text` — the chunk text
- `embedding` — the dense vector
- `metadata` — source, page, timestamp, content hash

> **Key Takeaway:** Ingestion takes raw documents → loads them → splits into chunks → removes duplicates → converts to vectors → outputs `ProcessedChunk` objects ready for the vector store.

---

## 4. Vector Stores — The Memory of RAG

### What a Vector Store Is

A **vector store** is a database optimized for one specific query: *"Given this vector, find the most similar vectors in the database."*

Regular databases are great for exact matches (`WHERE name = 'Alice'`). Vector stores are great for *similarity* matches (`find the 5 most semantically similar documents to this query`).

> **Analogy: A Library with Magic Index Cards**
>
> Remember the librarian who creates index cards? A vector store is like a shelf of index cards where the cards are arranged so that cards describing *similar topics* are physically next to each other. When you hand the librarian a new card, they walk to the nearest matching cards and pull them off the shelf.

### ChromaDB: Why It Was Chosen

DeepResearch AI uses **ChromaDB** for three reasons:

1. **Persistent** — Data survives restarts (uses `PersistentClient`, not in-memory)
2. **Local** — Runs on your machine, no cloud account needed
3. **Easy** — Simple Python API, great for getting started

ChromaDB uses an **HNSW index** (Hierarchical Navigable Small World) — a graph-based data structure that makes approximate nearest-neighbor search very fast, even with millions of vectors.

```python
# From src/vectorstore/chroma_store.py
class ChromaStore:
    def __init__(self, collection_name: str = "research_documents") -> None:
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine distance
        )
```

### The VectorStoreProtocol (`src/vectorstore/base.py`)

Every vector store backend must implement this interface:

```python
# From src/vectorstore/base.py
class VectorStoreProtocol(Protocol):
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]: ...

    async def upsert(self, chunks: list[ProcessedChunk]) -> int: ...
    async def delete(self, ids: list[str]) -> int: ...
```

This is the **contract** between the vector store and the rest of the system. If you wanted to swap ChromaDB for Pinecone, you'd just implement this protocol — no other code changes needed.

### ChromaDB Score Conversion

ChromaDB returns **cosine distance** (0 = identical, 2 = opposite). The project converts this to a **similarity score** (1 = identical, 0 = opposite):

```python
# From src/vectorstore/chroma_store.py
score=max(0.0, 1.0 - dist / 2.0),
```

### VectorStoreManager (`src/vectorstore/manager.py`)

The manager uses a **factory pattern** + **singleton** to ensure only one store instance exists:

```python
# From src/vectorstore/manager.py
_store: VectorStoreProtocol | None = None

def get_store() -> VectorStoreProtocol:
    global _store
    if _store is not None:
        return _store

    store_type = settings.vector_store_type
    if store_type == VectorStoreType.CHROMA:
        _store = ChromaStore()
    elif store_type == VectorStoreType.PINECONE:
        _store = PineconeStore()  # Stub for future use
    return _store
```

> **Key Takeaway:** The vector store is ChromaDB running locally with persistent storage. It stores vectors with an HNSW index for fast similarity search. The `VectorStoreProtocol` interface makes it swappable.

---

## 5. The Retrieval Engine — Finding Relevant Information

This is the **core** of the RAG system. Retrieval is responsible for finding the most relevant chunks for a user's query. DeepResearch AI uses a multi-stage approach.

### 5a. Dense Retrieval (`src/retrieval/dense.py`)

**How it works:**
1. Convert the user's query into a vector using the same embedding model used for documents
2. Search the vector store for the nearest neighbors using cosine similarity
3. Return the top-k results

```python
# From src/retrieval/dense.py
class DenseRetriever:
    def __init__(self, vector_store: VectorStoreProtocol) -> None:
        self._vector_store = vector_store

    async def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        return await self._vector_store.search(query_embedding, top_k=top_k)
```

**Cosine similarity** measures the angle between two vectors. If two vectors point in the same direction (meaning they represent similar content), the angle is 0° and similarity is 1.0.

**When dense retrieval works well:**
- Semantic matching: "revenue growth" matches "financial expansion"
- Conceptual queries: "How does the system handle errors?" matches "error handling mechanisms"

**When dense retrieval fails:**
- Exact keyword matching: searching for "ACME-2024-Q3" might miss a document that says "ACME 2024 Q3 Report" (different formatting)
- Rare terms: unusual proper nouns, technical codes, product IDs

### 5b. Sparse Retrieval / BM25 (`src/retrieval/sparse.py`)

**BM25** is a classic information retrieval algorithm from the 1990s. It scores documents based on how often query terms appear, weighted by how rare those terms are across the entire corpus.

> **Analogy: Keyword Highlighter**
>
> If dense retrieval is like a librarian who understands *meaning*, BM25 is like a librarian who highlights *exact keywords*. Each has strengths the other lacks.

**How BM25 works:**
1. **Tokenize** the query: lowercase + split on non-alphanumeric characters
2. **Score** each document: `TF × IDF` — term frequency (how often the word appears) × inverse document frequency (how rare the word is overall)
3. **Rank** by score, return top-k

```python
# From src/retrieval/sparse.py
def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return [t for t in re.split(r"\W+", text.lower()) if t]

class SparseRetriever:
    def __init__(self, documents: list[str], metadatas: list[ChunkMetadata]) -> None:
        tokenized = [_tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenized)  # Build the index at init time

    async def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [SearchResult(...) for idx in ranked_indices]
```

**Why BM25 still matters:** Dense retrieval understands meaning but can miss exact terms. BM25 catches exact matches that dense retrieval misses. Together, they cover each other's blind spots.

### 5c. Hybrid Retrieval with RRF (`src/retrieval/hybrid.py`)

Neither dense nor sparse retrieval alone is perfect. **Reciprocal Rank Fusion (RRF)** merges the two ranked lists into one.

> **Analogy: Two Librarians Voting**
>
> Imagine two librarians independently rank 20 books for your question. RRF combines their rankings by giving points based on rank position. Books ranked #1 by either librarian get lots of points. Books ranked #20 get very few. The final ranking is the point total.

**The RRF Formula:**

```
RRF_score(d) = Σ  1 / (k + rank_i(d))
                i∈{dense, sparse}

where:
  d = a document chunk
  rank_i(d) = the rank of d in retrieval method i (1-indexed)
  k = smoothing constant (default: 60)
```

**Why k=60?** From the original 2009 paper by Cormack, Clarke, and Buettcher: k=60 was found to be near-optimal in pilot experiments. The constant "mitigates the impact of high rankings by outlier systems" — meaning it prevents a single retrieval method from dominating.

**Step-by-step example:**

```
Query: "What is the chunking strategy?"

Dense retrieval ranks:
  Rank 1: Chunk A ("Recursive splitting with chunk_size=512...")
  Rank 2: Chunk C ("Semantic splitting uses embeddings...")
  Rank 5: Chunk B ("BM25 tokenizes on non-alphanumeric...")

Sparse retrieval ranks:
  Rank 1: Chunk B ("BM25 tokenizes on non-alphanumeric...")
  Rank 3: Chunk A ("Recursive splitting with chunk_size=512...")
  Rank 4: Chunk C ("Semantic splitting uses embeddings...")

RRF scores (k=60):
  Chunk A: 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226
  Chunk B: 1/(60+5) + 1/(60+1) = 0.01538 + 0.01639 = 0.03177
  Chunk C: 1/(60+2) + 1/(60+4) = 0.01613 + 0.01563 = 0.03176

Final ranking: Chunk A > Chunk B > Chunk C
```

```python
# From src/retrieval/hybrid.py
class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, k: int = 60) -> None:
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._k = k

    def _rrf_score(self, rank: int) -> float:
        return 1.0 / (self._k + rank)
```

> **Key Takeaway:** RRF merges dense and sparse results using a simple scoring formula. k=60 prevents any single retrieval method from dominating.

### 5d. Cross-Encoder Reranking (`src/retrieval/reranker.py`)

After hybrid fusion, results are "good but not perfect." A **cross-encoder** reranks them with much higher precision.

**Bi-encoder vs. Cross-encoder:**

```
Bi-encoder (used in dense retrieval — FAST):
  Query → [Encoder] → query_vector
  Doc   → [Encoder] → doc_vector
  score = cosine(query_vector, doc_vector)
  ✓ Pre-compute all doc vectors offline
  ✓ Search millions in milliseconds
  ✗ Query never "sees" the document during encoding

Cross-encoder (used in reranking — PRECISE):
  [CLS] Query [SEP] Document [SEP] → [Transformer] → score
  ✓ Query and document tokens attend to each other
  ✓ Understands exact word matches, negation, nuance
  ✗ Must process each (query, doc) pair separately
  ✗ Too slow for millions of documents
```

**The retrieve-then-rerank pattern:**
1. **Bi-encoder** retrieves top 50-100 candidates (fast, approximate)
2. **Cross-encoder** reranks those candidates to the top 5-10 (slow, precise)

```python
# From src/retrieval/reranker.py
class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model = CrossEncoder(model_name)

    async def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        pairs = [[query, r.text] for r in results]
        scores = self._model.predict(pairs)
        scored = sorted(zip(results, scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [SearchResult(id=r.id, text=r.text, metadata=r.metadata, score=float(s)) for r, s in scored]
```

> **Key Takeaway:** Cross-encoders are slow but precise. They read the query and document together, unlike bi-encoders which process them separately.

### 5e. Multi-Query Expansion (`src/retrieval/multi_query.py`)

Users ask one question, but the answer might be phrased differently in documents. **Multi-query expansion** uses an LLM to generate alternative phrasings.

```python
# From src/retrieval/multi_query.py
_DEFAULT_PROMPT = (
    "You are a helpful research assistant. Generate {n} alternative phrasings "
    "of the user's question that capture different aspects. Return ONLY the "
    "rewritten queries, one per line.\n\nOriginal: {query}"
)

class MultiQueryExpander:
    def expand(self, query: str, num_queries: int = 3) -> list[str]:
        if self._llm is None:
            return [query]  # Fallback: return original only
        prompt = _DEFAULT_PROMPT.format(n=num_queries, query=query)
        response = self._llm.invoke(prompt)
        return [query] + cleaned_rewrites[:num_queries]
```

**Example:**
```
Original: "What chunking strategy does the project use?"
Generated:
  1. "How does the system split documents into pieces?"
  2. "What text splitting approach is configured?"
  3. "Describe the document chunking configuration."
```

If no LLM is configured, the expander gracefully falls back to using only the original query.

### 5f. The Retrieval Pipeline Orchestrator (`src/retrieval/pipeline.py`)

The `RetrievalPipeline` wires everything together into one flow:

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       ↓
┌──────────────┐
│ Multi-Query  │  Generate N alternative phrasings
│ Expansion    │
└──────┬───────┘
       ↓
┌──────────────┐    ┌──────────────┐
│ Dense        │    │ Sparse       │
│ Retrieval    │    │ (BM25)       │
└──────┬───────┘    └──────┬───────┘
       ↓                   ↓
┌──────────────────────────────────┐
│         RRF Fusion               │  Merge ranked lists
└──────────────┬───────────────────┘
               ↓
┌──────────────────┐
│ Cross-Encoder    │  Re-score with precision
│ Reranking        │
└──────────┬───────┘
           ↓
┌──────────────────┐
│ RetrievalResult  │  Ranked chunks + metadata
└──────────────────┘
```

```python
# From src/retrieval/pipeline.py
class RetrievalPipeline:
    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        # 1. Multi-query expansion
        queries = self._expander.expand(query) if self._enable_multi_query else [query]

        # 2. Dense retrieval
        query_embedding = await self._embedder.embed_query(query)
        dense_results = await self._dense.retrieve(query_embedding, top_k=top_k * 2)

        # 3. Sparse retrieval
        sparse_results = await self._sparse.retrieve(query, top_k=top_k * 2) if self._sparse else []

        # 4. RRF fusion
        if sparse_results:
            hybrid = HybridRetriever(self._dense, self._sparse)
            fused = await hybrid.retrieve(query, query_embedding, top_k=top_k)
        else:
            fused = dense_results[:top_k]

        # 5. Rerank
        if self._reranker and fused:
            fused = await self._reranker.rerank(query, fused, top_k=top_k)

        return RetrievalResult(query=query, expanded_queries=queries, results=fused, ...)
```

> **Key Takeaway:** The retrieval pipeline is a multi-stage process: expand query → retrieve via dense + sparse → fuse with RRF → rerank with cross-encoder → return ranked results.

---

## 6. Generation — Turning Retrieved Context into Answers

### The Generation Pipeline (`src/agents/generation.py`)

Once retrieval finds the best chunks, the generation pipeline assembles them into a prompt and asks the LLM to answer.

```python
# From src/agents/generation.py
class GenerationPipeline:
    async def generate_answer(self, query: str, retrieval_result: RetrievalResult) -> dict:
        context = self._build_context(retrieval_result)     # Format chunks
        messages = self._build_messages(query, context)      # Build prompt
        answer = await self._llm.generate(messages)          # Call LLM
        return {"answer": answer, "sources": sources, "query": query}

    @staticmethod
    def _build_context(result: RetrievalResult) -> str:
        """Format retrieved chunks into a numbered context string."""
        parts = []
        for idx, sr in enumerate(result.results, 1):
            parts.append(f"[Source {idx}] {sr.text}")
        return "\n\n".join(parts) if parts else "No relevant context found."
```

### LLM Provider Abstraction (`src/agents/llm_provider.py`)

The project supports both OpenAI and Anthropic behind a single interface:

```python
# From src/agents/llm_provider.py
class LLMProvider:
    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        self._provider = provider or self._infer_provider(settings.llm_model)
        self._model = model or settings.llm_model

    @staticmethod
    def _infer_provider(model: str) -> str:
        """Heuristic: gpt-* → openai, everything else → anthropic."""
        if model.startswith("gpt") or model.startswith("o1"):
            return "openai"
        return "anthropic"
```

Changing providers requires only a config change — no code changes.

### Conversation Memory (`src/agents/memory.py`)

A sliding window of recent messages keeps the conversation context without exceeding token limits:

```python
# From src/agents/memory.py
class ConversationMemory:
    def __init__(self, window_size: int = 5) -> None:
        self._window_size = window_size
        self._messages: list[dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self._window_size:
            self._messages = self._messages[-self._window_size:]  # Keep last N
```

### System Prompts (`src/agents/prompts.py`)

The prompt engineering is critical for grounded, cited responses:

```python
# From src/agents/prompts.py
RESEARCH_PROMPT = (
    "You are a research assistant. Answer the user's question using ONLY the "
    "provided context. If the context does not contain enough information to "
    "answer the question, say that you cannot answer based on the available "
    "sources. When you use information from a source, cite it inline using "
    "[Source N] notation where N is the 1-based index of the source."
)
```

The key instruction is **"Answer using ONLY the provided context."** This prevents the LLM from hallucinating information not found in the retrieved documents.

> **Key Takeaway:** Generation takes retrieved chunks → formats them as numbered sources → builds a prompt instructing the LLM to answer from only those sources → returns a cited answer.

---

## 7. The API Layer

### FastAPI Endpoints (`src/api/main.py`)

The REST API exposes three endpoints:

```
GET  /                    → Service info
GET  /api/v1/health       → Health check
POST /api/v1/ingest       → Upload a document for ingestion
POST /api/v1/query        → Ask a question
```

**The query flow:**

```python
# From src/api/routes/query.py
@router.post("/api/v1/query")
async def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    store = get_store()                                      # 1. Get vector store
    retrieval = RetrievalPipeline(vector_store=store)        # 2. Set up retrieval
    result = await retrieval.retrieve(request.query)          # 3. Retrieve chunks
    generator = GenerationPipeline()                         # 4. Set up generation
    gen_result = await generator.generate_answer(request.query, result)  # 5. Generate
    return QueryResponse(answer=gen_result["answer"], sources=sources)
```

**Middleware** handles CORS (allowing the Streamlit UI on port 8501 to call the API) and request logging.

```python
# From src/api/middleware.py
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info("HTTP request", method=request.method, path=request.url.path,
                     status_code=response.status_code, latency_ms=round(elapsed_ms, 2))
        return response
```

> **Key Takeaway:** The API is a thin layer: `/ingest` saves documents to the vector store, `/query` runs the full retrieval → generation pipeline.

---

## 8. The Streamlit UI

### Chat Interface (`src/ui/pages/research_chat.py`)

The Streamlit UI provides two pages:

1. **💬 Research Chat** — Ask questions and get cited answers
2. **📚 Knowledge Base** — View and manage ingested documents

**How the chat works:**

```python
# From src/ui/pages/research_chat.py
def render() -> None:
    st.title("💬 Research Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        display_message(msg["role"], msg["content"])

    # Handle new queries
    if query := st.chat_input("Ask a research question..."):
        display_message("user", query)
        with st.spinner("Searching knowledge base..."):
            result = _send_query(query)
        if result:
            display_message("assistant", result["answer"])
            display_sources(result["sources"])
```

The UI sends requests to the FastAPI backend at `http://localhost:8000`.

> **Key Takeaway:** The Streamlit UI connects to the FastAPI backend, providing a chat interface for querying and a file upload for ingesting documents.

---

## 9. Configuration & Settings

### Pydantic Settings (`src/config/settings.py`)

All configuration comes from environment variables (or a `.env` file):

```python
# From src/config/settings.py
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM Provider Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Vector Store
    vector_store_type: str = "chroma"
    chroma_persist_dir: str = "./data/chroma_db"

    # Model Configuration
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k: int = 5

settings = Settings()
```

### The `.env` File Pattern

```bash
# .env (never committed to git)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

Key settings to know:

| Setting | Default | Purpose |
|---------|---------|---------|
| `chunk_size` | 512 | Max characters per chunk |
| `chunk_overlap` | 64 | Shared characters between chunks |
| `embedding_model` | text-embedding-3-small | Which embedding model to use |
| `llm_model` | gpt-4o-mini | Which LLM for generation |
| `top_k` | 5 | How many chunks to retrieve |
| `vector_store_type` | chroma | Backend: chroma or pinecone |

> **Key Takeaway:** All configuration lives in `.env` + Pydantic Settings. No secrets in source code. Change providers via config, not code.

---

## 10. Key Concepts Glossary

| Term | Definition |
|------|-----------|
| **Embedding** | A list of numbers (vector) that captures the semantic meaning of text. Created by an embedding model. |
| **Vector** | A list of floats (e.g., `[0.12, -0.34, 0.56, ...]`) representing a point in high-dimensional space. |
| **Cosine Similarity** | A metric measuring the angle between two vectors. 1.0 = identical meaning, 0.0 = unrelated. |
| **Chunk** | A small piece of text (typically 512 characters) split from a larger document for storage and retrieval. |
| **Token** | The smallest unit a language model processes. Roughly 4 characters or ¾ of a word in English. |
| **Context Window** | The maximum number of tokens an LLM can process in a single request (e.g., 128K tokens for GPT-4). |
| **RRF** | Reciprocal Rank Fusion — a formula for merging multiple ranked lists into one unified ranking. |
| **BM25** | A classic keyword-based search algorithm using term frequency and inverse document frequency. |
| **Cross-Encoder** | A model that processes a (query, document) pair together for precise relevance scoring. Slow but accurate. |
| **Bi-Encoder** | A model that encodes query and document separately into vectors. Fast but approximate. |
| **HNSW** | Hierarchical Navigable Small World — a graph index enabling fast approximate nearest-neighbor search. |
| **PersistentClient** | ChromaDB's mode that saves data to disk, surviving restarts. |
| **LangChain** | A Python framework for building applications powered by language models. Provides loaders, splitters, embeddings. |
| **LangGraph** | LangChain's framework for building stateful, multi-step agent workflows with conditional routing. |
| **CrewAI** | A framework for orchestrating multiple AI agents working together on a task. |
| **Guardrails** | Input/output validation layers that filter, fix, or reject unsafe/irrelevant content. |
| **RAGAS** | Retrieval-Augmented Generation Assessment — a framework for evaluating RAG system quality. |
| **DeepEval** | An evaluation framework for testing LLM outputs against ground truth. |
| **QLoRA** | Quantized Low-Rank Adaptation — a technique for fine-tuning large models efficiently on consumer GPUs. |
| **GGUF** | GPT-Generated Unified Format — a file format for running quantized models locally (e.g., via Ollama). |

---

## 11. Architecture Diagram

```
                            ┌─────────────────────────────────────────┐
                            │              STREAMLIT UI               │
                            │  ┌──────────────┐  ┌────────────────┐  │
                            │  │ Research Chat │  │ Knowledge Base │  │
                            │  └──────┬───────┘  └────────┬───────┘  │
                            └─────────┼───────────────────┼──────────┘
                                      │ POST /query        │ POST /ingest
                                      ↓                    ↓
                            ┌─────────────────────────────────────────┐
                            │           FASTAPI SERVER                 │
                            │  ┌──────────┐  ┌──────────┐            │
                            │  │  /query  │  │ /ingest  │  /health   │
                            │  └────┬─────┘  └────┬─────┘            │
                            └───────┼──────────────┼──────────────────┘
                                    ↓              ↓
                    ┌───────────────┐      ┌───────────────┐
                    │  RETRIEVAL    │      │  INGESTION    │
                    │  PIPELINE     │      │  PIPELINE     │
                    │               │      │               │
                    │ ┌───────────┐ │      │ ┌───┐ ┌─────┐│
                    │ │Multi-Query│ │      │ │Ld │ │Chnk ││
                    │ │ Expansion │ │      │ └─┬─┘ └──┬──┘│
                    │ └─────┬─────┘ │      │   ↓      ↓   │
                    │       ↓       │      │ ┌───┐ ┌─────┐│
                    │ ┌───────────┐ │      │ │Ded│ │Emb  ││
                    │ │  Dense +  │ │      │ └───┘ └──┬──┘│
                    │ │  Sparse   │ │      └──────────┼───┘
                    │ └─────┬─────┘ │                 ↓
                    │       ↓       │      ┌──────────────────┐
                    │ ┌───────────┐ │      │   CHROMA DB      │
                    │ │   RRF     │ │      │  (Persistent)    │
                    │ │  Fusion   │ │←─────│  HNSW Index      │
                    │ └─────┬─────┘ │      │  Vector Store    │
                    │       ↓       │      └──────────────────┘
                    │ ┌───────────┐ │
                    │ │Rerank(CE) │ │
                    │ └─────┬─────┘ │
                    └───────┼───────┘
                            ↓
                    ┌───────────────┐
                    │  GENERATION   │
                    │  PIPELINE     │
                    │               │
                    │ ┌───────────┐ │
                    │ │  Context  │ │
                    │ │ Assembly  │ │
                    │ └─────┬─────┘ │
                    │       ↓       │
                    │ ┌───────────┐ │
                    │ │ LLM Call  │ │ ← OpenAI / Anthropic
                    │ │(Grounded) │ │
                    │ └─────┬─────┘ │
                    │       ↓       │
                    │ ┌───────────┐ │
                    │ │  Source   │ │
                    │ │  Citations│ │
                    │ └───────────┘ │
                    └───────────────┘
                            ↓
                    ┌───────────────┐
                    │    RESPONSE   │
                    │               │
                    │ "According to │
                    │  [Source 1]   │
                    │  and [Source 3]│
                    │  ..."        │
                    └───────────────┘
```

---

## 12. What's Next (Phase 2-4 Preview)

### Phase 2: Multi-Agent Orchestration

Replace the single-chain RAG pipeline with a team of specialized AI agents:

- **Query Analyst** — Decomposes complex questions into sub-questions
- **Researcher** — Executes searches and gathers evidence
- **Fact-Checker** — Verifies claims against sources
- **Synthesizer** — Combines findings into coherent answers
- **Critic** — Reviews quality and flags issues

Orchestrated via **LangGraph** (state machine) and **CrewAI** (agent definitions).

### Phase 3: Safety Layer & Evaluation

- **Input guards** — Filter inappropriate or adversarial queries
- **Output guards** — Validate factual accuracy and source attribution
- **RAGAS evaluation** — Automated quality benchmarks (faithfulness >0.85, relevancy >0.90)
- **DeepEval** — Compare against golden datasets

### Phase 4: Fine-Tuning & Self-Improvement

- **Dataset curation** — Collect high-scoring Q&A pairs from evaluations
- **QLoRA training** — Fine-tune Mistral 7B on domain-specific data
- **A/B evaluation** — Compare base vs. fine-tuned model
- **GGUF export** — Deploy the fine-tuned model via Ollama

---

> **Final Note:** This guide is meant to be your companion as you explore the DeepResearch AI codebase. Start with the ingestion pipeline, follow the data through the vector store, and watch the retrieval engine find relevant information. The best way to learn is to trace a single query from the UI all the way to the response. Happy researching!
