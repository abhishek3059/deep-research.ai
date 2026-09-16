# Lesson 1: Document Ingestion Pipeline

**Objective:** Understand why and how raw documents are transformed into searchable, deduplicated, embedded chunks ready for a RAG (Retrieval-Augmented Generation) system.

---

## 1. What is Document Ingestion?

In a RAG system, the model cannot search the internet or access your files directly. Instead, you **preprocess** your documents into a searchable knowledge base. Think of it like organizing a library:

- **Raw documents** = Boxes of unsorted books dumped at the library entrance.
- **Ingestion** = The librarian who opens each box, catalogues each book, splits long chapters into sections, assigns unique IDs, removes duplicate copies, and places them on shelves with a call number.

Only after this work can a patron (the user's query) quickly find the relevant shelf and retrieve the exact passage they need.

### Why Not Just Store the Whole Document?

1. **Context Windows:** LLMs have limited context. A 50-page PDF is too large to fit entirely into a prompt.
2. **Retrieval Precision:** Searching a 50-page document for a specific fact is like searching for a needle in a haystack. Smaller chunks improve precision.
3. **Cost Efficiency:** Embedding and storing large texts is expensive. Smaller, meaningful chunks reduce cost and latency.
4. **Deduplication:** The same paragraph might appear in multiple documents. Storing it once saves space and prevents biased retrieval.

---

## 2. The Pipeline Architecture

The ingestion pipeline in DeepResearch AI follows a four-stage sequence:

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   LOAD      │ → │   CHUNK     │ → │   DEDUP     │ → │   EMBED     │
│  (files)    │   │ (documents) │   │  (unique)   │   │ (vectors)   │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
```

### Stage 1: Load
**File:** `src/ingestion/loaders.py`

- **What:** Reads raw bytes from PDFs, text files, CSVs, Markdown, and web pages.
- **Why:** Different formats require different parsing logic (PDFs have pages, CSVs have rows, HTML has tags).
- **Implementation:** Uses LangChain's community loaders (`PyPDFLoader`, `TextLoader`, `CSVLoader`, `UnstructuredMarkdownLoader`, `WebBaseLoader`).
- **Key Design Decision:** Wraps synchronous loaders in `asyncio.to_thread` to keep the event loop free for other async operations.

### Stage 2: Chunk
**File:** `src/ingestion/chunker.py`

- **What:** Splits large documents into smaller, fixed-size pieces.
- **Why:** LLMs have context limits, and smaller chunks improve retrieval accuracy.
- **Implementation:** Supports two strategies via `ChunkStrategy` enum:
  - **Recursive Character Splitting** (default): Splits by characters, respecting paragraph and sentence boundaries. Fast, predictable.
  - **Semantic Splitting:** Uses embeddings to find natural semantic breaks. Higher quality but slower (requires embedding model).
- **Key Design Decision:** Configurable `chunk_size` (default 512 characters) and `chunk_overlap` (default 64 characters). Overlap ensures context continuity across chunk boundaries.

### Stage 3: Dedup
**File:** `src/ingestion/deduplicator.py`

- **What:** Removes duplicate chunks based on content hash (SHA-256).
- **Why:** The same text might appear in multiple documents or even within the same document. Duplicate chunks waste storage and embedding API calls, and can bias retrieval results.
- **Implementation:** Computes SHA-256 of each chunk's `page_content`. Keeps only the first occurrence of each unique hash.
- **Key Design Decision:** Runs **after** chunking but **before** embedding. This avoids paying to embed duplicate content.

### Stage 4: Embed
**File:** `src/ingestion/embedder.py`

- **What:** Converts text chunks into dense vector representations (embeddings).
- **Why:** Vector representations enable semantic search. Instead of matching keywords, you can find chunks that are *conceptually similar* to a query.
- **Implementation:** Uses OpenAI's embedding API (`OpenAIEmbeddings`). Processes chunks in batches of 100 to respect API limits.
- **Key Design Decision:** Model and API key come from `settings.embedding_model` and `settings.openai_api_key`. No hardcoded values.

---

## 3. Key Concepts Deep Dive

### Document Loaders

Different file formats encode information differently:

| Format | Loader | Granularity |
|--------|--------|-------------|
| PDF | `PyPDFLoader` | One document per page |
| Text | `TextLoader` | One document per file |
| CSV | `CSVLoader` | One document per row |
| Markdown | `UnstructuredMarkdownLoader` | Preserves headings and structure |
| Web | `WebBaseLoader` | Parses HTML into text |

**Why this matters:** A PDF loader that reads raw bytes would miss text in images or formatted tables. A CSV loader that reads as plain text would lose row-column relationships. Each loader knows the format's nuances.

**Example from code** (`loaders.py:86-96`):
```python
if source_type is SourceType.PDF:
    return await self.load_pdf(source)
if source_type is SourceType.TEXT:
    return await self.load_text(source)
# ...
```

### Chunking Strategies

#### Recursive Character Splitting
- **How:** Splits text by `\n\n` (paragraphs), then `\n` (lines), then spaces, then characters, until chunks are under the size limit.
- **When to use:** Default choice. Fast, deterministic, no external dependencies.
- **Example:** A 2000-character document with `chunk_size=512, chunk_overlap=64` produces ~4 chunks with 64-character overlaps.

#### Semantic Splitting
- **How:** Embeds each sentence, then finds breakpoints where adjacent sentences have low cosine similarity (i.e., topic shifts).
- **When to use:** When document structure is complex or you need semantically coherent chunks.
- **Trade-off:** ~3x slower than recursive (requires embedding API calls during chunking).

**Code reference** (`chunker.py:152-162`):
```python
if strategy is ChunkStrategy.RECURSIVE:
    return RecursiveCharacterTextSplitter(
        chunk_size=self._chunk_size,
        chunk_overlap=self._chunk_overlap,
    )
if strategy is ChunkStrategy.SEMANTIC:
    from langchain_experimental.text_splitter import SemanticChunker
    return SemanticChunker(embeddings)
```

### Embeddings

**What:** A fixed-length list of numbers (vector) that captures the semantic meaning of text. For example, "The cat sat on the mat" and "A feline rested on the rug" would have similar vectors, even though they share no common words.

**How they work:**
1. A pre-trained model (e.g., `text-embedding-3-small`) reads the text.
2. It outputs a vector of 1536 dimensions (for OpenAI's model).
3. Similar texts produce vectors that are close in Euclidean space.

**Why they matter for RAG:**
- **Semantic search:** Instead of matching exact keywords, you find chunks whose vectors are closest to the query vector.
- **Multilingual:** Some models understand multiple languages, enabling cross-lingual retrieval.

**Batch processing** (`embedder.py:96-98`):
```python
for start in range(0, len(texts), self._batch_size):
    batch = texts[start : start + self._batch_size]
    vectors.extend(await self._embed_batch(batch))
```

### Deduplication

**How:** SHA-256 hash of the chunk's text content. Same text → same hash → duplicate detected.

**Why content-hash dedup:**
- **Deterministic:** Same input always produces the same hash.
- **Collision-resistant:** Virtually impossible for different texts to produce the same hash.
- **Efficient:** Hashing is fast; comparing hashes is O(1) with a set.

**When it helps:**
- Ingesting overlapping documents (e.g., multiple versions of a paper).
- Re-ingesting a document that was partially updated.
- Cross-document overlap (e.g., a quoted paragraph appearing in multiple sources).

**Code reference** (`deduplicator.py:50-54`):
```python
digest = self.compute_hash(document.page_content)
if digest in seen:
    continue
seen.add(digest)
unique.append(document)
```

---

## 4. Code Walkthrough: The Pipeline Orchestrator

**File:** `src/ingestion/pipeline.py`

The `IngestionPipeline` class wires the four stages together:

```python
class IngestionPipeline:
    async def ingest(self, source: str, source_type: SourceType | None = None) -> list[ProcessedChunk]:
        # 1. Load
        documents = await self._loader.load(source, source_type)
        
        # 2. Chunk
        chunks = await self._chunker.achunk_recursive(documents)
        
        # 3. Dedup
        unique_chunks = self._deduplicator.deduplicate(chunks)
        
        # 4. Embed
        embeddings = await self._embedder.embed_documents(unique_chunks)
        
        # 5. Package
        processed = self._build_processed_chunks(unique_chunks, embeddings, source)
        return processed
```

**Key design decisions:**
1. **Async-first:** All I/O (loading, embedding) is async. CPU-bound work (chunking, hashing) is wrapped in `asyncio.to_thread`.
2. **Dependency injection:** Each stage accepts an optional implementation, enabling easy testing with mocks.
3. **Contract compliance:** The output `ProcessedChunk` dataclass matches the interface contract defined in `docs/CONTRACTS.md` section 4.1.
4. **Deterministic IDs:** Chunk IDs are computed as `SHA256(source:content_hash)`, ensuring the same chunk always gets the same ID across re-ingestion.

---

## 5. Common Pitfalls

### Chunk Size Too Large
- **Symptom:** Retrieval returns huge blocks of text, wasting context window space.
- **Fix:** Reduce `chunk_size` (e.g., 256–512 characters).

### Chunk Size Too Small
- **Symptom:** Chunks lack context; the model can't answer questions that require surrounding information.
- **Fix:** Increase `chunk_size` or increase `chunk_overlap`.

### No Overlap
- **Symptom:** Information split across chunk boundaries is lost. For example, a sentence that starts in chunk 1 and ends in chunk 2.
- **Fix:** Set `chunk_overlap` to ~10–20% of `chunk_size` (e.g., 64 for 512).

### Embedding Model Mismatch
- **Symptom:** Documents embedded with one model are searched with a different model. Vectors are incompatible, retrieval fails silently.
- **Fix:** Always use the same embedding model for ingestion and querying. Store the model name in metadata.

### Dedup Before Chunking
- **Symptom:** Two different chunks from the same paragraph are both kept, but they're semantically redundant.
- **Fix:** Run dedup **after** chunking (as implemented). Each chunk is deduplicated independently.

### Ignoring Empty Chunks
- **Symptom:** Chunks with only whitespace or newlines pollute the vector store.
- **Fix:** Filter them out. The pipeline already does this (`chunker.py:137`):
  ```python
  filtered = [chunk for chunk in chunks if chunk.page_content.strip()]
  ```

---

## 6. Summary

The ingestion pipeline transforms raw documents into a structured, searchable knowledge base:

1. **Load** – Parse diverse file formats into LangChain `Document` objects.
2. **Chunk** – Split documents into small, overlapping pieces for precise retrieval.
3. **Dedup** – Remove redundant chunks to save storage and embedding costs.
4. **Embed** – Convert text to dense vectors for semantic search.

The pipeline is **async**, **configurable**, and **contract-driven**—each stage has a clear input/output interface, making it easy to swap implementations (e.g., switch from OpenAI to HuggingFace embeddings) without changing the orchestrator.

**Next steps:** Once chunks are embedded, they flow into the vector store (ChromaDB or Pinecone) and are retrieved by the retrieval pipeline when a user asks a question.

---

*Lesson created: 2026-09-13 | DeepResearch AI Project*
