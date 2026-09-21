# Nirvexa AI Engine — Phase 4: Document Ingestion, Knowledge Management & Hybrid Retrieval

## 1. Overview & Architectural Objective

Phase 4 extends the Nirvexa AI Engine beyond public web search into **private, user-isolated document intelligence** and **hybrid grounded reasoning**.

The primary objectives of Phase 4 are:
1. **Document Ingestion:** Enable authenticated users to upload private documents (`.pdf`, `.txt`, `.md`) with strict file type validation and size limits (5MB).
2. **Text Parsing & Chunking:** Extract text cleanly using pre-installed libraries (`pdfplumber` for PDFs, standard decoders for text/markdown) and chunk content using the Phase 2 deterministic `TextChunker`.
3. **Relational Knowledge Persistence:** Store document metadata and chunked evidence in SQLAlchemy models (`AIDocument` and `AIDocumentChunk`) with database-level multi-tenant user isolation.
4. **User-Isolated Document Retrieval:** Search across private document chunks using the Phase 2 `KeywordReranker` while ensuring User A can never access or cite User B's documents.
5. **Hybrid Retrieval Mode:** Unify live web search and private document chunks into a single fused candidate pool, ranked neutrally and packaged into a budgeted `EvidencePack`.
6. **Grounded Citations with Document Provenance:** Produce verifiable answers citing both web URLs (`[S1]`) and uploaded document titles (`[S2] Document.pdf`).

---

## 2. Directory Structure & Files Created

```
nirvexa-backend/
├── app/
│   ├── models/
│   │   └── ai_document.py                 # SQLAlchemy models: AIDocument, AIDocumentChunk
│   ├── ai_engine/
│   │   ├── coordinator.py                 # WebResearchEngine supporting search_mode ('web', 'document', 'hybrid')
│   │   ├── documents/
│   │   │   ├── __init__.py                # Package exports
│   │   │   ├── schemas.py                 # Pydantic schemas (DocumentSummary, DocumentUploadResponse, etc.)
│   │   │   ├── parser.py                  # DocumentParser: PDF (pdfplumber), TXT, Markdown
│   │   │   └── service.py                 # DocumentService: Ingestion, quotas, multi-tenant isolation, deletion
│   │   ├── retrieval/schemas.py           # EvidenceRequest with search_mode
│   │   └── reasoning/schemas.py           # AnswerRequest with search_mode
│   └── routes/
│       └── ai_engine.py                   # POST /upload, GET /documents, DELETE /documents/<id>
├── docs/
│   └── AI_ENGINE_PHASE_4.md               # Technical specification
└── tests/
    └── test_ai_engine_phase4.py           # 16 unit, integration, and security tests
```

---

## 3. Data Models & Database Design

### `ai_documents`
Represents an uploaded document owned by a user:
- `id` (`String(36)`): Primary Key (UUID)
- `user_id` (`String(36)`): Foreign Key (`users.id`), Indexed
- `filename` (`String(255)`): Original uploaded filename
- `title` (`String(255)`): Document display title
- `file_type` (`String(20)`): File extension (`pdf`, `txt`, `md`)
- `file_size` (`Integer`): File size in bytes
- `chunk_count` (`Integer`): Number of generated chunks
- `created_at` / `updated_at`: UTC timestamps

### `ai_document_chunks`
Represents a granular chunk extracted from a document:
- `id` (`String(64)`): Primary Key (UUID/hash)
- `document_id` (`String(36)`): Foreign Key (`ai_documents.id`, On Delete Cascade), Indexed
- `user_id` (`String(36)`): Foreign Key (`users.id`), Indexed (denormalized for high-performance scoped filtering)
- `chunk_index` (`Integer`): Ordinal position in document
- `text` (`Text`): Clean chunk text
- `char_count` (`Integer`): Character length
- `metadata_json` (`JSON`): Page numbers, offsets, document title

---

## 4. Ingestion & Document Parser

- **`DocumentParser`** (`app/ai_engine/documents/parser.py`):
  - Validates file extensions against whitelist: `.pdf`, `.txt`, `.md`.
  - Enforces magic bytes check for PDFs (`%PDF-`).
  - Extracts text from PDFs page-by-page using `pdfplumber`.
  - Normalizes line breaks (`\r\n` -> `\n`) and strips null bytes (`\x00`).
  - Rejects empty files or PDFs with no extractable text.

---

## 5. Multi-Tenant User Isolation & Quota Management

- **`DocumentService`** (`app/ai_engine/documents/service.py`):
  - **Ownership Guarantee:** Every query and deletion filters strictly on `user_id == current_user_id`.
  - **ID Enumeration Defense:** Attempting to retrieve or delete documents belonging to another user returns `404 Not Found`.
  - **Cascading Deletion:** Deleting a document explicitly purges all associated chunks from `ai_document_chunks`.
  - **Quota Limits (Configurable & Clamped):**
    - `AI_ENGINE_MAX_DOC_SIZE_BYTES = 5242880` (5MB maximum per file)
    - `AI_ENGINE_MAX_DOCS_PER_USER = 10` (Maximum 10 documents per user)
    - `AI_ENGINE_MAX_DOC_CHUNKS_PER_USER = 250` (Maximum 250 chunks per user)

---

## 6. Hybrid Retrieval & Coordinator Flow

`WebResearchEngine` (`app/ai_engine/coordinator.py`) supports three search modes:

1. **`mode="web"`:** Executes Phase 1 web research, extracts and normalizes URLs, chunks content, and builds an `EvidencePack`.
2. **`mode="document"`:** Bypasses live web requests completely. Retrieves user-scoped document chunks from `ai_document_chunks` and builds a document-backed `EvidencePack`.
3. **`mode="hybrid"`:** Fuses candidates from both web research and user private documents into a combined candidate list. The Phase 2 `KeywordReranker` scores and ranks them neutrally, packaging the top results within the context budget into a unified `EvidencePack`.

Downstream Phase 3 `ReasoningEngine`, `PromptBuilder`, and `CitationValidator` consume the unified `EvidencePack` seamlessly without contract modifications.

---

## 7. API Reference

### 1. Upload Document
```http
POST /api/ai/documents/upload
Authorization: Bearer <JWT_ACCESS_TOKEN>
Content-Type: multipart/form-data

file: <binary_data> (.pdf, .txt, .md)
title: "Quarterly Strategy" (optional)
```
**Response (201 Created):**
```json
{
  "document": {
    "id": "doc_a1b2c3d4",
    "user_id": "usr_123",
    "filename": "strategy.pdf",
    "title": "Quarterly Strategy",
    "file_type": "pdf",
    "file_size": 1048576,
    "chunk_count": 8,
    "created_at": "2026-09-21T14:00:00Z"
  },
  "message": "Document uploaded and indexed successfully."
}
```

### 2. List Documents
```http
GET /api/ai/documents
Authorization: Bearer <JWT_ACCESS_TOKEN>
```
**Response (200 OK):**
```json
{
  "documents": [...],
  "total_documents": 2,
  "total_chunks": 14
}
```

### 3. Delete Document
```http
DELETE /api/ai/documents/<document_id>
Authorization: Bearer <JWT_ACCESS_TOKEN>
```
**Response (200 OK):**
```json
{
  "document_id": "doc_a1b2c3d4",
  "message": "Document deleted successfully."
}
```

### 4. Hybrid Grounded Answering
```http
POST /api/ai/research/answer
Authorization: Bearer <JWT_ACCESS_TOKEN>
Content-Type: application/json

{
  "query": "What are our compliance deadlines and how do they compare with industry standards?",
  "search_mode": "hybrid",
  "max_results": 5
}
```
**Response (200 OK):** Returns grounded answer text citing both document (`[S1]`) and web (`[S2]`) evidence items.

---

## 8. Verification & Test Coverage

All 85 unit and integration tests pass with zero external network requests:
- 1 ElevenLabs test
- 26 Phase 1 tests (`tests/test_ai_engine_phase1.py`)
- 24 Phase 2 tests (`tests/test_ai_engine_phase2.py`)
- 18 Phase 3 tests (`tests/test_ai_engine_phase3.py`)
- 16 Phase 4 tests (`tests/test_ai_engine_phase4.py`)
