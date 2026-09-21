"""
tests/test_ai_engine_phase4.py
Comprehensive unit, integration, and security test suite for AI Engine Phase 4:
Document Ingestion, User-Isolated Knowledge Base, and Hybrid Retrieval Layer.
ZERO external network calls.
"""
import io
import pytest
from unittest.mock import MagicMock, patch

from app import create_app
from app.extensions import db
from app.models.user import User
from app.models.ai_document import AIDocument, AIDocumentChunk
from app.ai_engine.coordinator import WebResearchEngine
from app.ai_engine.documents.parser import DocumentParser, DocumentParserError
from app.ai_engine.documents.service import DocumentService, QuotaExceededError
from app.ai_engine.reasoning.engine import ReasoningEngine
from app.ai_engine.reasoning.providers.mock import MockLLMProvider
from app.ai_engine.retrieval.evidence import EvidenceBuilder
from app.ai_engine.schemas.research import SearchRequest, ResearchResponse, WebSearchResult
from app.ai_engine.search.provider import MockSearchProvider
from app.ai_engine.search.service import SearchService


@pytest.fixture
def app():
    """Create Flask test application with in-memory DB and test configuration."""
    test_app = create_app("testing")
    test_app.config["AI_ENGINE_ENABLED"] = True
    test_app.config["AI_ENGINE_LLM_ENABLED"] = True
    test_app.config["AI_ENGINE_LLM_PROVIDER"] = "mock"
    test_app.config["AI_ENGINE_DOCUMENTS_ENABLED"] = True
    test_app.config["RATELIMIT_ENABLED"] = False
    test_app.config["AI_ENGINE_MAX_DOCS_PER_USER"] = 3
    test_app.config["AI_ENGINE_MAX_DOC_CHUNKS_PER_USER"] = 15

    with test_app.app_context():
        AIDocument.__table__.create(db.engine, checkfirst=True)
        AIDocumentChunk.__table__.create(db.engine, checkfirst=True)
        yield test_app
        db.session.remove()
        AIDocumentChunk.__table__.drop(db.engine, checkfirst=True)
        AIDocument.__table__.drop(db.engine, checkfirst=True)


@pytest.fixture
def client(app):
    return app.test_client()


# ==============================================================================
# 1. DocumentParser Tests
# ==============================================================================

def test_parser_txt_and_md():
    """Verifies that plain text and markdown files are parsed cleanly."""
    txt_content = b"Nirvexa is an AI platform.\nIt assists engineers with career transitions."
    doc = DocumentParser.parse(txt_content, "about.txt")

    assert doc.title == "about"
    assert doc.file_type == "txt"
    assert "Nirvexa is an AI platform." in doc.text
    assert doc.char_count == len(doc.text)

    md_content = b"# Architecture Overview\n\n- Component A\n- Component B\n"
    md_doc = DocumentParser.parse(md_content, "arch.md")
    assert md_doc.file_type == "md"
    assert "Component A" in md_doc.text


def test_parser_unsupported_format():
    """Verifies rejection of unsupported extensions."""
    with pytest.raises(DocumentParserError, match="Unsupported file format"):
        DocumentParser.parse(b"binary data", "malicious.exe")


def test_parser_empty_file():
    """Verifies rejection of empty byte streams."""
    with pytest.raises(DocumentParserError, match="Uploaded file is empty"):
        DocumentParser.parse(b"", "empty.txt")


def test_parser_invalid_pdf_header():
    """Verifies rejection of files with .pdf extension lacking %PDF- magic bytes."""
    with pytest.raises(DocumentParserError, match="Invalid PDF header"):
        DocumentParser.parse(b"This is not a real PDF", "fake.pdf")


def test_parser_valid_pdf_mocked():
    """Verifies PDF extraction workflow using mock pdfplumber."""
    fake_pdf_bytes = b"%PDF-1.4 Fake PDF stream"

    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Page 1: System Requirements and Specifications."
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Page 2: Security and Threat Model Analysis."

    mock_pdf_instance = MagicMock()
    mock_pdf_instance.pages = [mock_page1, mock_page2]
    mock_pdf_instance.__enter__.return_value = mock_pdf_instance
    mock_pdf_instance.__exit__.return_value = None

    with patch("pdfplumber.open", return_value=mock_pdf_instance):
        doc = DocumentParser.parse(fake_pdf_bytes, "manual.pdf")
        assert doc.file_type == "pdf"
        assert "System Requirements" in doc.text
        assert "Threat Model Analysis" in doc.text
        assert doc.metadata["total_pages"] == 2


# ==============================================================================
# 2. DocumentService & Quota Tests
# ==============================================================================

def test_document_service_upload_and_chunk(app):
    """Verifies document upload, chunking, and database persistence."""
    with app.app_context():
        service = DocumentService()
        sample_text = (
            "FastAPI is a modern, fast web framework for building APIs with Python.\n\n"
            "It is based on Starlette for the web parts and Pydantic for the data parts.\n\n"
            "FastAPI provides automatic interactive API documentation with Swagger UI."
        ).encode("utf-8")

        doc = service.upload_document(
            user_id="user_alice",
            file_bytes=sample_text,
            filename="fastapi_overview.txt",
            title="FastAPI Guide",
        )

        assert doc.id is not None
        assert doc.user_id == "user_alice"
        assert doc.title == "FastAPI Guide"
        assert doc.chunk_count >= 1

        # Verify chunks stored in DB
        chunks = AIDocumentChunk.query.filter_by(document_id=doc.id).all()
        assert len(chunks) == doc.chunk_count
        assert all(ch.user_id == "user_alice" for ch in chunks)
        assert any("Starlette" in ch.text for ch in chunks)


def test_document_service_max_docs_quota(app):
    """Verifies enforcement of AI_ENGINE_MAX_DOCS_PER_USER quota."""
    with app.app_context():
        service = DocumentService()
        # Upload up to max (3 docs in test fixture)
        for i in range(3):
            service.upload_document(
                user_id="user_alice",
                file_bytes=f"Document content {i}".encode("utf-8"),
                filename=f"doc_{i}.txt",
            )

        # 4th upload must raise QuotaExceededError
        with pytest.raises(QuotaExceededError, match="User document limit reached"):
            service.upload_document(
                user_id="user_alice",
                file_bytes=b"Overflow document",
                filename="overflow.txt",
            )


def test_document_service_user_isolation(app):
    """Verifies strict multi-tenant isolation between user documents."""
    with app.app_context():
        service = DocumentService()

        # Alice uploads a private document
        doc_alice = service.upload_document(
            user_id="user_alice",
            file_bytes=b"Alice secret project specifications and roadmap.",
            filename="alice_secret.txt",
        )

        # Bob uploads a private document
        doc_bob = service.upload_document(
            user_id="user_bob",
            file_bytes=b"Bob internal quarterly salary review metrics.",
            filename="bob_salary.txt",
        )

        # Alice's document listing only contains Alice's documents
        alice_docs = service.list_documents(user_id="user_alice")
        assert len(alice_docs) == 1
        assert alice_docs[0].id == doc_alice.id

        # Bob cannot access or delete Alice's document
        assert service.get_document(user_id="user_bob", document_id=doc_alice.id) is None
        assert service.delete_document(user_id="user_bob", document_id=doc_alice.id) is False

        # Alice's document is still intact
        assert service.get_document(user_id="user_alice", document_id=doc_alice.id) is not None

        # Query candidates for Alice only retrieve Alice's chunks
        alice_candidates = service.retrieve_candidates_for_query(user_id="user_alice", query="secret roadmap")
        assert len(alice_candidates) >= 1
        assert all(c.source_id == f"doc_{doc_alice.id}" for c in alice_candidates)
        assert not any("salary" in c.content.lower() for c in alice_candidates)


def test_document_deletion_cascades_chunks(app):
    """Verifies that deleting a document removes all associated chunks."""
    with app.app_context():
        service = DocumentService()
        doc = service.upload_document(
            user_id="user_alice",
            file_bytes=b"Chunk 1 text.\n\nChunk 2 text.\n\nChunk 3 text.",
            filename="cascade_test.txt",
        )
        doc_id = doc.id
        assert AIDocumentChunk.query.filter_by(document_id=doc_id).count() > 0

        # Delete document
        deleted = service.delete_document(user_id="user_alice", document_id=doc_id)
        assert deleted is True
        assert AIDocument.query.filter_by(id=doc_id).first() is None
        assert AIDocumentChunk.query.filter_by(document_id=doc_id).count() == 0


# ==============================================================================
# 3. Hybrid Retrieval & Coordinator Integration Tests
# ==============================================================================

def test_coordinator_search_modes(app):
    """Verifies WebResearchEngine behavior across 'web', 'document', and 'hybrid' modes."""
    with app.app_context():
        mock_search = MockSearchProvider()
        search_service = SearchService(provider=mock_search)
        document_service = DocumentService()
        mock_llm = MockLLMProvider(
            default_response="Synthesized answer citing evidence items [S1] and [S2]."
        )
        reasoning_engine = ReasoningEngine(provider=mock_llm)

        engine = WebResearchEngine(
            search_service=search_service,
            reasoning_engine=reasoning_engine,
            document_service=document_service,
        )

        # Alice uploads a private document about distributed systems
        document_service.upload_document(
            user_id="user_alice",
            file_bytes=b"Raft is a consensus algorithm designed as an alternative to Paxos.",
            filename="raft_consensus.txt",
            title="Raft Algorithm Notes",
        )

        req = SearchRequest(query="consensus algorithm", max_results=3, fetch_content=False)

        # 1. Mode 'web': returns only web sources
        pack_web = engine.research_evidence(req, search_mode="web", user_id="user_alice")
        assert all(item.source_type == "web" for item in pack_web.items)

        # 2. Mode 'document': returns only document sources
        pack_doc = engine.research_evidence(req, search_mode="document", user_id="user_alice")
        assert len(pack_doc.items) >= 1
        assert all(item.source_type == "document" for item in pack_doc.items)
        assert any("Raft" in item.text for item in pack_doc.items)

        # 3. Mode 'hybrid': fuses both web and document sources
        pack_hybrid = engine.research_evidence(req, search_mode="hybrid", user_id="user_alice")
        source_types = {item.source_type for item in pack_hybrid.items}
        assert "document" in source_types
        assert "web" in source_types

        # 4. End-to-end grounded answer in hybrid mode
        answer_resp = engine.research_and_answer(req, search_mode="hybrid", user_id="user_alice")
        assert answer_resp.grounding_status == "grounded"
        assert len(answer_resp.citations) >= 1


# ==============================================================================
# 4. API Endpoint Tests
# ==============================================================================

class TestDocumentEndpoints:

    def test_upload_endpoint_unauthorized(self, client):
        """Verifies 401 when Authorization header is missing."""
        resp = client.post("/api/ai/documents/upload", data={})
        assert resp.status_code == 401

    def test_upload_endpoint_missing_file(self, app, client):
        """Verifies 400 when multipart file is omitted."""
        with patch("app.middleware.auth_middleware.decode_token", return_value={"sub": "user_alice", "type": "access"}):
            resp = client.post(
                "/api/ai/documents/upload",
                data={},
                headers={"Authorization": "Bearer valid-token"},
            )
            assert resp.status_code == 400
            data = resp.get_json()
            assert data["error"] == "validation_error"

    def test_upload_endpoint_success(self, app, client):
        """Verifies 201 Created on successful document upload."""
        file_content = io.BytesIO(b"PostgreSQL supports advanced indexing and MVCC concurrency.")
        with patch("app.middleware.auth_middleware.decode_token", return_value={"sub": "user_alice", "type": "access"}):
            resp = client.post(
                "/api/ai/documents/upload",
                data={
                    "file": (file_content, "postgres_guide.txt"),
                    "title": "Postgres Guide",
                },
                content_type="multipart/form-data",
                headers={"Authorization": "Bearer valid-token"},
            )
            assert resp.status_code == 201
            data = resp.get_json()
            assert data["document"]["filename"] == "postgres_guide.txt"
            assert data["document"]["title"] == "Postgres Guide"
            assert data["document"]["chunk_count"] >= 1

    def test_list_documents_endpoint(self, app, client):
        """Verifies GET /api/ai/documents lists only authenticated user documents."""
        with app.app_context():
            DocumentService().upload_document(
                user_id="user_alice",
                file_bytes=b"Alice test doc",
                filename="alice_test.txt",
            )

        with patch("app.middleware.auth_middleware.decode_token", return_value={"sub": "user_alice", "type": "access"}):
            resp = client.get(
                "/api/ai/documents",
                headers={"Authorization": "Bearer valid-token"},
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["total_documents"] == 1
            assert data["documents"][0]["filename"] == "alice_test.txt"

    def test_delete_document_endpoint(self, app, client):
        """Verifies DELETE /api/ai/documents/<doc_id>."""
        with app.app_context():
            doc = DocumentService().upload_document(
                user_id="user_alice",
                file_bytes=b"Doc to delete",
                filename="delete_me.txt",
            )
            doc_id = doc.id

        with patch("app.middleware.auth_middleware.decode_token", return_value={"sub": "user_alice", "type": "access"}):
            # Delete as Alice succeeds
            resp = client.delete(
                f"/api/ai/documents/{doc_id}",
                headers={"Authorization": "Bearer valid-token"},
            )
            assert resp.status_code == 200

            # Deleting again returns 404
            resp_again = client.delete(
                f"/api/ai/documents/{doc_id}",
                headers={"Authorization": "Bearer valid-token"},
            )
            assert resp_again.status_code == 404

    def test_answer_endpoint_hybrid_mode(self, app, client):
        """Verifies POST /api/ai/research/answer supports search_mode='hybrid'."""
        mock_web_research = ResearchResponse(
            query="explain consensus protocol",
            results=[
                WebSearchResult(
                    title="Consensus Protocols",
                    url="https://example.com/consensus",
                    snippet="Consensus protocols ensure distributed systems agree on values.",
                    content="Consensus protocols ensure distributed systems agree on values.",
                    source="example.com",
                )
            ],
            total_results=1,
            search_status="success",
        )

        with patch("app.middleware.auth_middleware.decode_token", return_value={"sub": "user_alice", "type": "access"}), \
             patch("app.routes.ai_engine._engine.research", return_value=mock_web_research):
            resp = client.post(
                "/api/ai/research/answer",
                json={
                    "query": "explain consensus protocol",
                    "search_mode": "hybrid",
                    "max_results": 3,
                },
                headers={"Authorization": "Bearer valid-token"},
            )
            # Even with no docs uploaded, hybrid falls back gracefully to web
            assert resp.status_code == 200
            data = resp.get_json()
            assert "answer" in data
            assert data["query"] == "explain consensus protocol"

    def test_evidence_endpoint_document_and_hybrid_modes(self, app, client):
        """Verifies POST /api/ai/research/evidence supports 'document' and 'hybrid' modes."""
        with app.app_context():
            DocumentService().upload_document(
                user_id="user_alice",
                file_bytes=b"Kubernetes orchestrates containerized workloads across node clusters.",
                filename="k8s.txt",
                title="K8s Notes",
            )

        mock_web_research = ResearchResponse(
            query="Kubernetes orchestration",
            results=[
                WebSearchResult(
                    title="Kubernetes Official",
                    url="https://kubernetes.io",
                    snippet="Kubernetes is an open-source container orchestration system.",
                    content="Kubernetes is an open-source container orchestration system.",
                    source="kubernetes.io",
                )
            ],
            total_results=1,
            search_status="success",
        )

        with patch("app.middleware.auth_middleware.decode_token", return_value={"sub": "user_alice", "type": "access"}), \
             patch("app.routes.ai_engine._engine.research", return_value=mock_web_research):
            # Document mode
            resp_doc = client.post(
                "/api/ai/research/evidence",
                json={
                    "query": "Kubernetes orchestration",
                    "search_mode": "document",
                },
                headers={"Authorization": "Bearer valid-token"},
            )
            assert resp_doc.status_code == 200
            data_doc = resp_doc.get_json()
            assert len(data_doc["items"]) >= 1
            assert all(item["source_type"] == "document" for item in data_doc["items"])

            # Hybrid mode
            resp_hybrid = client.post(
                "/api/ai/research/evidence",
                json={
                    "query": "Kubernetes orchestration",
                    "search_mode": "hybrid",
                },
                headers={"Authorization": "Bearer valid-token"},
            )
            assert resp_hybrid.status_code == 200
            data_hybrid = resp_hybrid.get_json()
            source_types = {item["source_type"] for item in data_hybrid["items"]}
            assert "document" in source_types
            assert "web" in source_types

    def test_documents_disabled_feature_flag(self, app, client):
        """Verifies 503 behavior when AI_ENGINE_DOCUMENTS_ENABLED is False."""
        app.config["AI_ENGINE_DOCUMENTS_ENABLED"] = False

        with patch("app.middleware.auth_middleware.decode_token", return_value={"sub": "user_alice", "type": "access"}):
            # 1. Document upload
            resp = client.post(
                "/api/ai/documents/upload",
                data={"file": (io.BytesIO(b"data"), "test.txt")},
                content_type="multipart/form-data",
                headers={"Authorization": "Bearer valid-token"},
            )
            assert resp.status_code == 503
            assert resp.get_json()["error"] == "ai_engine_documents_disabled"

            # 2. Document listing
            resp = client.get("/api/ai/documents", headers={"Authorization": "Bearer valid-token"})
            assert resp.status_code == 503
            assert resp.get_json()["error"] == "ai_engine_documents_disabled"

            # 3. Document deletion
            resp = client.delete("/api/ai/documents/any-id", headers={"Authorization": "Bearer valid-token"})
            assert resp.status_code == 503
            assert resp.get_json()["error"] == "ai_engine_documents_disabled"

            # 4. Evidence endpoint with document mode
            resp = client.post(
                "/api/ai/research/evidence",
                json={"query": "test", "search_mode": "document"},
                headers={"Authorization": "Bearer valid-token"},
            )
            assert resp.status_code == 503
            assert resp.get_json()["error"] == "ai_engine_documents_disabled"

            # 5. Answer endpoint with hybrid mode
            resp = client.post(
                "/api/ai/research/answer",
                json={"query": "test", "search_mode": "hybrid"},
                headers={"Authorization": "Bearer valid-token"},
            )
            assert resp.status_code == 503
            assert resp.get_json()["error"] == "ai_engine_documents_disabled"

    def test_ai_engine_master_disabled_feature_flag(self, app, client):
        """Verifies 503 ai_engine_disabled on document endpoints when AI_ENGINE_ENABLED is False."""
        app.config["AI_ENGINE_ENABLED"] = False

        with patch("app.middleware.auth_middleware.decode_token", return_value={"sub": "user_alice", "type": "access"}):
            # Document upload
            resp = client.post(
                "/api/ai/documents/upload",
                data={"file": (io.BytesIO(b"data"), "test.txt")},
                content_type="multipart/form-data",
                headers={"Authorization": "Bearer valid-token"},
            )
            assert resp.status_code == 503
            assert resp.get_json()["error"] == "ai_engine_disabled"

            # Document listing
            resp = client.get("/api/ai/documents", headers={"Authorization": "Bearer valid-token"})
            assert resp.status_code == 503
            assert resp.get_json()["error"] == "ai_engine_disabled"

            # Document deletion
            resp = client.delete("/api/ai/documents/any-id", headers={"Authorization": "Bearer valid-token"})
            assert resp.status_code == 503
            assert resp.get_json()["error"] == "ai_engine_disabled"
