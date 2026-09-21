# Nirvexa AI Engine — Phase 5: PostgreSQL + pgvector Semantic Retrieval & Hybrid Fusion

## 1. Overview & Architectural Objective

Phase 5 introduces **high-performance vector embeddings, semantic nearest-neighbor retrieval, and multi-modal Reciprocal Rank Fusion (RRF)** to the Nirvexa AI Engine, building directly on top of the Phase 4 private document foundation.

The primary objectives of Phase 5 are:
1. **Zero New External Dependencies:** Utilize native PostgreSQL `vector` capabilities without requiring `pip install pgvector`. Implement a dialect-aware SQLAlchemy `TypeDecorator` (`VectorType`) that maps to native PostgreSQL `vector(dim)` in production and compiles cleanly to `Text`/`JSON` in SQLite/testing environments.
2. **Pluggable Embedding Providers:** Support industry standard embedding architectures via a unified `BaseEmbeddingProvider` interface:
   - `MockEmbeddingProvider`: Zero-network, deterministic SHA-256 unit vectors for rapid CI/CD and testing.
   - `GeminiEmbeddingProvider`: Google `text-embedding-004` (768-dim) using pre-existing `google-generativeai`.
   - `OpenAICompatibleEmbeddingProvider`: OpenAI `text-embedding-3-small` (1536-dim) or compatible endpoints using pre-existing `openai`.
   - `OllamaEmbeddingProvider`: Local self-hosted embeddings (`nomic-embed-text`) using pre-existing `httpx`.
3. **Automatic Ingestion Embedding:** Batch-embed document chunks during ingestion in `DocumentService.upload_document()`, gracefully falling back to lexical-only storage if the embedding provider is temporarily unavailable.
4. **Dialect-Adaptive Semantic Search:**
   - **PostgreSQL:** Executes in-database cosine distance queries using native pgvector `<=>` operators with HNSW index acceleration.
   - **SQLite / In-Memory:** Seamlessly falls back to in-memory cosine similarity calculation across stored chunk vectors.
5. **Reciprocal Rank Fusion (RRF):** Combine lexical retrieval (Phase 4 `KeywordReranker`) and semantic retrieval into a single, balanced candidate ranking without arbitrary score weighting.
6. **Robust Multi-Tenant Isolation:** Ensure all vector similarity searches strictly filter by `user_id == current_user_id`.
7. **Offline Bounded Backfill Utility:** Provide an idempotent, batch-processing script (`backfill_chunk_embeddings`) to populate vector embeddings for pre-existing or failed chunks without blocking API transactions.

---

## 2. Directory Structure & Files Created / Modified

```
nirvexa-backend/
├── app/
│   ├── models/
│   │   └── ai_document.py                 # VectorType TypeDecorator + embedding column on AIDocumentChunk
│   ├── ai_engine/
│   │   ├── documents/
│   │   │   └── service.py                 # Hybrid retrieval (semantic + lexical + RRF) and batch embedding
│   │   ├── embeddings/
│   │   │   ├── __init__.py                # Package exports
│   │   │   ├── base.py                    # BaseEmbeddingProvider abstract interface & exception hierarchy
│   │   │   ├── mock.py                    # MockEmbeddingProvider (deterministic, zero network)
│   │   │   ├── gemini.py                  # GeminiEmbeddingProvider (text-embedding-004)
│   │   │   ├── openai.py                  # OpenAICompatibleEmbeddingProvider (text-embedding-3-small)
│   │   │   ├── ollama.py                  # OllamaEmbeddingProvider (nomic-embed-text)
│   │   │   ├── factory.py                 # Configuration-driven EmbeddingProviderFactory
│   │   │   └── backfill.py                # Safe, bounded offline backfill script
│   │   └── retrieval/
│   │       └── fusion.py                  # Reciprocal Rank Fusion (RRF) implementation
├── migrations/versions/
│   └── e7b9c1d2e3f4_add_vector_embedding_to_ai_document_chunks.py # Reversible pgvector migration + HNSW index
├── config.py                              # Phase 5 semantic retrieval configuration settings
├── docs/
│   └── AI_ENGINE_PHASE_5.md               # Phase 5 architecture & specification document
└── tests/
    ├── test_ai_engine_phase4.py           # Updated fixture with mock LLM provider
    └── test_ai_engine_phase5.py           # 20 comprehensive unit, integration, and security tests
```

---

## 3. Data Models & Vector Persistence

### Dialect-Aware `VectorType`
Implemented in `app/models/ai_document.py`:
- Inherits from `sqlalchemy.types.TypeDecorator`.
- For PostgreSQL: compiles as `vector(dimension)`.
- For SQLite / other dialects: compiles as `Text` and serializes/deserializes Python `List[float]` to and from JSON.

### Database Schema Updates (`ai_document_chunks`)
- Added column: `embedding = db.Column(VectorType(dimension=768), nullable=True)`
- Indexing: In PostgreSQL, indexed via HNSW:
  ```sql
  CREATE INDEX IF NOT EXISTS ix_ai_document_chunks_embedding_hnsw
  ON ai_document_chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
  ```

---

## 4. Embedding Providers & Architecture

All providers extend `BaseEmbeddingProvider` (`app/ai_engine/embeddings/base.py`):
```python
class BaseEmbeddingProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_text(self, text: str) -> List[float]: ...

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]: ...
```

### Provider Matrix

| Provider | Implementation | Model | Dimension | Dependencies |
| :--- | :--- | :--- | :--- | :--- |
| **`mock`** | `MockEmbeddingProvider` | `mock-embed-v1` | Configurable (default 768) | None (SHA-256 unit float) |
| **`gemini`** | `GeminiEmbeddingProvider` | `text-embedding-004` | 768 | `google-generativeai` |
| **`openai`** | `OpenAICompatibleEmbeddingProvider` | `text-embedding-3-small` | 1536 | `openai` |
| **`ollama`** | `OllamaEmbeddingProvider` | `nomic-embed-text` | 768 | `httpx` |

---

## 5. Reciprocal Rank Fusion (RRF)

Implemented in `app/ai_engine/retrieval/fusion.py`.

Given lexical candidate rankings $R_{lex}$ and semantic candidate rankings $R_{sem}$, RRF scores each unique candidate document $d$ according to:
$$RRF(d) = \sum_{m \in \{lex, sem\}} \frac{w_m}{k + r_m(d)}$$

Where:
- $k = 60$ (smoothing constant preventing top ranks from dominating).
- $r_m(d)$ is the 1-based rank of document $d$ in retrieval method $m$.
- $w_m$ is the modality weight (default $1.0$).

Benefits:
- Eliminates disparate score distribution issues between BM25/keyword scores and cosine similarity metrics.
- Boosts documents appearing in both lexical and semantic top-k queries.

---

## 6. Hybrid Retrieval Flow & Resilience

When a user query is received in `DocumentService.retrieve_candidates_for_query()`:
1. **Feature Flag Check:** If `AI_ENGINE_SEMANTIC_ENABLED` is `False`, executes standard Phase 4 lexical retrieval.
2. **Parallel Candidate Collection:**
   - **Semantic Search:** Generates query vector embedding and retrieves top semantic candidates using dialect-specific similarity.
   - **Lexical Search:** Executes Phase 4 substring & keyword matching.
3. **Graceful Fallback:** If embedding generation fails (e.g. rate-limiting, network timeout), logs a warning and falls back immediately to lexical-only search without raising a user-facing error.
4. **Fusion:** Runs Reciprocal Rank Fusion across semantic and lexical result sets.
5. **Context Budgeting:** Slices the top `AI_ENGINE_HYBRID_TOP_K` items for downstream packing into the `EvidencePack`.

---

## 7. Configuration Settings

| Setting | Default | Clamping | Description |
| :--- | :--- | :--- | :--- |
| `AI_ENGINE_SEMANTIC_ENABLED` | `False` | Boolean | Master switch for vector embedding & retrieval |
| `AI_ENGINE_EMBEDDING_PROVIDER` | `"gemini"` | `"mock"`, `"gemini"`, `"openai"`, `"ollama"` | Active embedding provider |
| `AI_ENGINE_EMBEDDING_MODEL` | `"text-embedding-004"` | String | Target embedding model identifier |
| `AI_ENGINE_EMBEDDING_DIMENSION` | `768` | `64` – `4096` | Target vector dimension |
| `AI_ENGINE_SEMANTIC_TOP_K` | `25` | `5` – `100` | Max candidates retrieved from vector search |
| `AI_ENGINE_HYBRID_TOP_K` | `30` | `5` – `100` | Max candidates returned after RRF fusion |

---

## 8. Verification & Test Suite

The Phase 5 test suite (`tests/test_ai_engine_phase5.py`) verifies all functional, resilience, security, and regression guarantees:
- **`test_01`–`test_05`:** Configuration clamping, factory instantiation, dimension validation, and provider error handling.
- **`test_06`–`test_09`:** Document embedding persistence, semantic vector similarity, multi-tenant isolation, and empty result handling.
- **`test_10`–`test_13`:** Reciprocal Rank Fusion, fallback on provider failure, deleted document exclusion, and bounded top-k slicing.
- **`test_14`:** Offline backfill batching utility.
- **`test_15`–`test_17`:** End-to-end coordinator and API endpoint integration (`/api/ai/research/evidence`, `/api/ai/research/answer`).
- **`test_18`–`test_20`:** Strict regression protection across Phase 1 web search, Phase 2 chunking/evidence building, Phase 3 reasoning/citations, and Phase 4 CRUD.

**Full Test Suite Run:**
- Total tests passed: **108 / 108**
- Zero live external network calls.
- `git diff --check`: Clean (0 whitespace/formatting errors).
