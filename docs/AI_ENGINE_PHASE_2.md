# Nirvexa AI Engine — Phase 2: Retrieval & Evidence Layer

## 1. Overview & Architectural Objective

Phase 2 builds directly upon the Phase 1 Web Research foundation to construct a robust, deterministic, and modular **Retrieval + Evidence Layer**.

The primary purpose of Phase 2 is to transform raw search and fetched web documents (as well as future internal Nirvexa database records and private enterprise documents) into clean, structured, scored, and traceable **Evidence Items** packaged in an **`EvidencePack`**. This packaged evidence serves as the verifiable, bounded context for downstream consumption by future Local LLMs / SLMs.

> [!IMPORTANT]
> **Scope Notice:** Phase 2 implements **only** the Retrieval → Normalization → Chunking → Reranking → Evidence Packaging pipeline. It does **not** perform local LLM inference, vector embedding model generation, or arbitrary crawler scraping.

### Master Architecture Position

```
User Query
    │
    ▼
Query Understanding (Future)
    │
    ▼
[Phase 1] Web Search / Nirvexa DB / Private Documents
    │
    ▼
[Phase 2] Retrieval Layer (Normalization, Deduplication, Quality Gate, Chunking)
    │
    ▼
[Phase 2] Reranker (Multi-Signal Keyword Scoring & Context Filtering)
    │
    ▼
[Phase 2] Evidence Pack (Structured, Traceable Context Budget)
    │
    ▼
[Phase 3] Local LLM / SLM (Future Inference)
    │
    ▼
Structured Answer with Authoritative Citations
```

---

## 2. Directory Structure & Files Created

All Phase 2 logic is isolated under `app/ai_engine/retrieval/`, integrated into `app/ai_engine/coordinator.py`, exposed internally in `app/routes/ai_engine.py`, configured in `config.py`, verified in `tests/test_ai_engine_phase2.py`, and documented here:

```
nirvexa-backend/
├── app/
│   ├── ai_engine/
│   │   ├── coordinator.py                 # WebResearchEngine.research_evidence() orchestrator
│   │   ├── retrieval/
│   │   │   ├── __init__.py                # Package exports
│   │   │   ├── schemas.py                 # Source-neutral schemas (RetrievalResult, EvidenceItem, EvidencePack)
│   │   │   ├── chunking.py                # Deterministic semantic TextChunker with sliding window
│   │   │   ├── service.py                 # RetrievalService (ingestion, normalizer, quality filter)
│   │   │   ├── evidence.py                # EvidenceBuilder (packaging, context budget, citations)
│   │   │   └── reranker/
│   │   │       ├── __init__.py            # Reranker exports
│   │   │       ├── base.py                # BaseReranker abstract interface
│   │   │       └── keyword.py             # Multi-signal KeywordReranker
│   │   ├── search/                        # (Phase 1 untouched)
│   │   ├── fetch/                         # (Phase 1 untouched)
│   │   ├── extraction/                    # (Phase 1 untouched)
│   │   └── schemas/                       # (Phase 1 untouched)
│   └── routes/
│       └── ai_engine.py                   # Internal POST /api/ai/research/evidence endpoint
├── config.py                              # Phase 2 context budget environment variables
├── tests/
│   ├── test_ai_engine_phase1.py           # Phase 1 unit test suite (100% passing)
│   └── test_ai_engine_phase2.py           # Phase 2 20-scenario test suite (100% passing)
└── docs/
    ├── AI_ENGINE_PHASE_1.md               # Phase 1 specifications
    └── AI_ENGINE_PHASE_2.md               # This document
```

---

## 3. Source-Neutral Contract Schemas

Phase 2 establishes a uniform data contract supporting multiple heterogeneous knowledge sources:

### `RetrievalResult`
Source-neutral document representation:
- `source_id: str` — Canonical identifier (normalized URL, document UUID, or DB primary key).
- `source_type: Literal["web", "nirvexa_db", "document", "knowledge", "unknown"]` — Origin type.
- `title: str` — Document title.
- `url: Optional[str]` — Source URL if originating from the web.
- `content: str` — Full clean body text.
- `snippet: Optional[str]` — Brief summary or search snippet.
- `metadata: Dict[str, Any]` — Provenance telemetry (domain, author, publication date, etc.).
- `retrieval_score: float` — Initial search rank score [1.0, 0.5, 0.33, ...].
- `relevance_score: float` — Post-reranking relevance score [0.0, 1.0].
- `timestamp: str` — ISO 8601 ingestion timestamp.

### `EvidenceItem`
Granular, verifiable evidence chunk strictly linked to a parent source:
- `evidence_id: str` — Deterministic unique ID (`evi_<sha256(source_id:chunk_index:text[:48])>`).
- `source_id: str` — Foreign key back to originating parent document.
- `text: str` — Extracted chunk content.
- `title: str` — Originating document title for citation.
- `url: Optional[str]` — Source URL for link generation.
- `source_type: SourceType` — Origin source type.
- `relevance_score: float` — Normalized relevance score [0.0, 1.0].
- `metadata: Dict[str, Any]` — Positional metadata (`chunk_index`, `total_chunks`, `char_count`, `start_char`, `end_char`).

### `SourceSummary`
Authoritative citation summary per source contributing evidence:
- `source_id: str`
- `title: str`
- `url: Optional[str]`
- `source_type: str`
- `chunk_count: int`

### `EvidencePack`
Bounded evidence payload delivered downstream:
- `query: str` — Original research query.
- `items: List[EvidenceItem]` — Filtered, ranked, and budgeted evidence chunks.
- `total_candidates: int` — Total document candidates ingested.
- `selected_items: int` — Final count of selected evidence chunks.
- `total_characters: int` — Cumulative character count within context budget.
- `generated_at: str` — ISO 8601 generation timestamp.
- `source_summary: List[SourceSummary]` — Distinct source citations for UI or LLM references.

---

## 4. Chunking Engine (`TextChunker`)

The chunker decomposes raw document text along natural syntactic and semantic boundaries:
1. **Paragraph & Sentence Boundaries:** Prefers splitting on double newlines (`\n\n`), then sentence terminators (`. `, `! `, `? `).
2. **Sliding-Window Overlap:** Configurable overlap (default 100 chars) preserves contextual continuity across adjacent chunks without cutting words in half.
3. **Oversized Block Protection:** Unformatted single-block text exceeding `max_chunk_chars` is safely partitioned along whitespace boundaries without crashing or memory bloat.
4. **Deterministic Telemetry:** Computes `chunk_index`, `total_chunks`, `start_char`, and `end_char` for exact provenance tracing.

---

## 5. Relevance Ranking (`KeywordReranker`)

To guarantee fast execution and zero heavy external ML dependencies in Phase 2, `KeywordReranker` implements a multi-signal baseline bounded mathematically in `[0.0, 1.0]`:

| Signal | Contribution | Description |
|---|---|---|
| **Exact Phrase Match** | +0.25 to +0.35 | Verbatim query substring match in title or body text. |
| **Title Token Coverage** | +0.00 to +0.25 | Ratio of content query tokens present in the document title. |
| **Text Token Coverage** | +0.00 to +0.30 | Ratio of content query tokens present in the chunk body text. |
| **Snippet Token Coverage** | +0.00 to +0.10 | Ratio of query tokens present in the source snippet. |
| **Term Frequency Density** | +0.00 to +0.10 | Logarithmically scaled occurrence frequency of query tokens relative to chunk length. |
| **Micro-Chunk Penalty** | 0.5x multiplier | Applied to chunks shorter than 40 characters to discount noise fragments. |

### Deterministic & Stable Sorting
Ties in relevance score are deterministically broken using:
`(-relevance_score, -len(text), evidence_id)`
Guaranteeing 100% reproducible ordering regardless of input sequence.

### Low-Relevance Filter
Chunks scoring below `min_relevance_score` (default 0.10) are discarded before context budgeting.

---

## 6. Context Budget & Configuration

All context constraints are safely configured in `config.py` with runtime fallback clamping to protect memory and token limits:

| Environment Variable | Type | Default | Clamped Range | Description |
|---|---|---|---|---|
| `AI_ENGINE_MAX_EVIDENCE_ITEMS` | int | `10` | 1 – 50 | Maximum number of evidence chunks in an EvidencePack. |
| `AI_ENGINE_MAX_EVIDENCE_CHARS` | int | `12000` | 500 – 100,000 | Maximum cumulative characters across all selected chunks. |
| `AI_ENGINE_MAX_CHUNK_CHARS` | int | `1000` | 100 – 5,000 | Maximum character size of an individual chunk. |
| `AI_ENGINE_CHUNK_OVERLAP_CHARS` | int | `100` | 0 – 500 | Sliding-window overlap between adjacent chunks. |
| `AI_ENGINE_MAX_CANDIDATES` | int | `20` | 1 – 100 | Maximum candidate documents ingested prior to chunking. |
| `AI_ENGINE_MIN_RELEVANCE_SCORE` | float | `0.10` | 0.0 – 1.0 | Minimum score threshold for an evidence chunk. |

---

## 7. Provenance & Citation Integrity

1. **No Orphan Chunks:** An `EvidenceItem` cannot be instantiated without a valid `source_id`.
2. **No Hallucinated Citations:** Citation metadata is derived strictly from verified candidate attributes extracted by Phase 1 (`title`, `canonical_url`, `source_type`).
3. **URL Normalization:** URLs are cleaned of query tracking artifacts (`utm_*`, `fbclid`, `gclid`) and trailing slashes to prevent duplicate citations for the same source.

---

## 8. Security & Network Isolation Guarantees

- **Zero Direct Fetching:** Phase 2 never initiates HTTP requests or socket connections. It operates exclusively on memory objects supplied by Phase 1 or internal sources.
- **SSRF & Size Protections Preserved:** All web documents must pass through Phase 1 SSRF filters, redirect limits, and the 2MB download cap.
- **Authentication & Rate Limiting:** The internal endpoint `POST /api/ai/research/evidence` enforces `@token_required`, `@limiter.limit("10 per minute")`, and checks `AI_ENGINE_ENABLED`.

---

## 9. API Endpoint Reference

### `POST /api/ai/research/evidence` (Internal / Development)

Transforms web research results into a structured, ranked `EvidencePack`.

#### Headers
- `Authorization: Bearer <access_token>`
- `Content-Type: application/json`

#### Request Body
```json
{
  "query": "fastapi high performance features",
  "max_results": 3,
  "max_evidence_items": 5,
  "max_evidence_chars": 6000
}
```

#### Response Body (`200 OK`)
```json
{
  "query": "fastapi high performance features",
  "total_candidates": 3,
  "selected_items": 3,
  "total_characters": 1845,
  "generated_at": "2026-09-21T11:00:00+00:00",
  "items": [
    {
      "evidence_id": "evi_8f7b2a19c4d3e5f6",
      "source_id": "https://fastapi.tiangolo.com/features",
      "title": "FastAPI Features",
      "url": "https://fastapi.tiangolo.com/features",
      "source_type": "web",
      "relevance_score": 0.925,
      "text": "FastAPI is based on Starlette and Pydantic, achieving performance on par with NodeJS and Go...",
      "metadata": {
        "chunk_index": 0,
        "total_chunks": 3,
        "char_count": 612
      }
    }
  ],
  "source_summary": [
    {
      "source_id": "https://fastapi.tiangolo.com/features",
      "title": "FastAPI Features",
      "url": "https://fastapi.tiangolo.com/features",
      "source_type": "web",
      "chunk_count": 2
    }
  ]
}
```

---

## 10. Future Reranker & Embedding Integration Roadmap

Phase 2 architecturally decouples the ranking mechanism via the abstract `BaseReranker` class:

```python
class BaseReranker(ABC):
    @abstractmethod
    def rank(self, query: str, items: List[EvidenceItem]) -> List[EvidenceItem]:
        pass
```

When Phase 3 or subsequent phases introduce local vector embedding or cross-encoder rerankers (e.g. BGE-Reranker, SentenceTransformers, or MiniLM):
1. Implement `EmbeddingReranker(BaseReranker)` in `app/ai_engine/retrieval/reranker/`.
2. Pass `EmbeddingReranker` to `EvidenceBuilder(reranker=EmbeddingReranker())`.
3. No changes to `WebResearchEngine`, `RetrievalService`, schemas, or API contracts are required.

---

## 11. Testing & Verification

A dedicated unit test suite covering all 20 required verification scenarios is located in `tests/test_ai_engine_phase2.py`:

```powershell
venv_new\Scripts\python.exe -m pytest tests/test_ai_engine_phase2.py -v
```

All 24 unit/API tests pass with zero external network connectivity.
All existing Phase 1 tests continue to pass with zero regressions.
