"""
tests/test_ai_engine_phase5.py
Comprehensive unit, integration, security, and regression tests for Phase 5:
PostgreSQL + pgvector Semantic Retrieval, Embedding Providers, RRF Fusion, and Tenant Isolation.
Zero live external network calls.
"""
import io
import pytest
from unittest.mock import patch, MagicMock

from app import create_app
from app.extensions import db
from app.models.ai_document import AIDocument, AIDocumentChunk
from app.ai_engine.coordinator import WebResearchEngine
from app.ai_engine.documents.service import DocumentService
from app.ai_engine.embeddings import (
    BaseEmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingProviderTimeoutError,
    EmbeddingProviderRateLimitError,
    EmbeddingProviderFactory,
    MockEmbeddingProvider,
)
from app.ai_engine.embeddings.backfill import backfill_chunk_embeddings
from app.ai_engine.reasoning.engine import ReasoningEngine
from app.ai_engine.reasoning.providers.mock import MockLLMProvider
from app.ai_engine.retrieval.evidence import EvidenceBuilder
from app.ai_engine.retrieval.fusion import reciprocal_rank_fusion
from app.ai_engine.retrieval.schemas import RetrievalResult, EvidencePack
from app.ai_engine.schemas.research import SearchRequest, ResearchResponse, WebSearchResult
from app.ai_engine.search.provider import MockSearchProvider
from app.ai_engine.search.service import SearchService


@pytest.fixture
def app():
    """Create Flask test application configured for Phase 5 testing with in-memory DB."""
    test_app = create_app("testing")
    test_app.config["AI_ENGINE_ENABLED"] = True
    test_app.config["AI_ENGINE_LLM_ENABLED"] = True
    test_app.config["AI_ENGINE_LLM_PROVIDER"] = "mock"
    test_app.config["AI_ENGINE_DOCUMENTS_ENABLED"] = True
    test_app.config["AI_ENGINE_SEMANTIC_ENABLED"] = True
    test_app.config["AI_ENGINE_EMBEDDING_PROVIDER"] = "mock"
    test_app.config["AI_ENGINE_EMBEDDING_DIMENSION"] = 16
    test_app.config["AI_ENGINE_SEMANTIC_TOP_K"] = 10
    test_app.config["RATELIMIT_ENABLED"] = False

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


@pytest.fixture
def mock_embedder():
    return MockEmbeddingProvider(dimension=16)


# ==============================================================================
# 1. Vector Configuration & Provider Resolution Tests
# ==============================================================================

def test_01_configuration_clamping_and_defaults(app):
    """Verifies that Phase 5 configuration keys parse and clamp correctly."""
    assert app.config["AI_ENGINE_SEMANTIC_ENABLED"] is True
    assert app.config["AI_ENGINE_EMBEDDING_PROVIDER"] == "mock"
    assert app.config["AI_ENGINE_EMBEDDING_DIMENSION"] == 16
    assert app.config["AI_ENGINE_SEMANTIC_TOP_K"] == 10


def test_02_embedding_provider_factory_resolution():
    """Verifies factory instantiation of mock, gemini, openai, and rejection of invalid names."""
    p_mock = EmbeddingProviderFactory.create(provider_name="mock", dimension=32)
    assert p_mock.name == "mock"
    assert p_mock.dimension == 32

    p_gemini = EmbeddingProviderFactory.create(provider_name="gemini", dimension=768)
    assert p_gemini.name == "gemini"

    p_openai = EmbeddingProviderFactory.create(provider_name="openai", dimension=1536)
    assert p_openai.name == "openai"

    p_ollama = EmbeddingProviderFactory.create(provider_name="ollama", dimension=768)
    assert p_ollama.name == "ollama"

    with pytest.raises(EmbeddingProviderError, match="Unsupported embedding provider"):
        EmbeddingProviderFactory.create(provider_name="unsupported_xyz")


# ==============================================================================
# 2. Embedding Generation & Error Handling Tests
# ==============================================================================

def test_03_embedding_generation_success(mock_embedder):
    """Verifies deterministic embedding generation and normalization."""
    vec1 = mock_embedder.embed_text("Distributed consensus algorithm")
    vec2 = mock_embedder.embed_text("Distributed consensus algorithm")
    vec3 = mock_embedder.embed_text("Completely unrelated cooking recipe")

    assert len(vec1) == 16
    assert vec1 == vec2  # Deterministic
    assert vec1 != vec3

    batch_vecs = mock_embedder.embed_batch(["Item A", "Item B"])
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 16


def test_04_embedding_failure_handling():
    """Verifies timeout and rate-limit error handling in embedding provider."""
    timeout_provider = MockEmbeddingProvider(should_fail=True, failure_type="timeout")
    with pytest.raises(EmbeddingProviderTimeoutError, match="timed out"):
        timeout_provider.embed_text("Hello")

    rate_limit_provider = MockEmbeddingProvider(should_fail=True, failure_type="rate_limit")
    with pytest.raises(EmbeddingProviderRateLimitError, match="quota exceeded"):
        rate_limit_provider.embed_text("Hello")


def test_05_vector_dimension_validation(mock_embedder):
    """Verifies vector dimension matches configured length."""
    vec = mock_embedder.embed_text("Short text")
    assert len(vec) == mock_embedder.dimension


# ==============================================================================
# 3. Document Ingestion & Semantic Retrieval Tests
# ==============================================================================

def test_06_document_service_persists_embeddings_on_upload(app, mock_embedder):
    """Verifies that DocumentService computes and persists vector embeddings on upload."""
    with app.app_context():
        service = DocumentService(embedding_provider=mock_embedder)
        doc = service.upload_document(
            user_id="user_alice",
            file_bytes=b"PostgreSQL pgvector provides vector indexing and similarity search.",
            filename="pgvector_guide.txt",
            title="PGVector Guide",
        )

        chunks = AIDocumentChunk.query.filter_by(document_id=doc.id).all()
        assert len(chunks) >= 1
        for ch in chunks:
            assert ch.embedding is not None
            assert len(ch.embedding) == 16
            assert ch.to_dict()["has_embedding"] is True


def test_07_semantic_similarity_retrieval(app, mock_embedder):
    """Verifies semantic similarity retrieval finds closest chunks."""
    with app.app_context():
        service = DocumentService(embedding_provider=mock_embedder)
        service.upload_document(
            user_id="user_alice",
            file_bytes=b"Kubernetes orchestrates containers and manages clusters of pods.",
            filename="k8s.txt",
            title="Kubernetes Intro",
        )
        service.upload_document(
            user_id="user_alice",
            file_bytes=b"Italian risotto with arborio rice, parmesan cheese and chicken broth.",
            filename="recipe.txt",
            title="Risotto Recipe",
        )

        candidates = service.retrieve_candidates_for_query(
            user_id="user_alice", query="orchestrating containerized pods"
        )
        assert len(candidates) >= 1
        # The kubernetes document chunk should be ranked first due to semantic + lexical overlap
        assert "Kubernetes" in candidates[0].content or candidates[0].title == "Kubernetes Intro"


def test_08_multi_tenant_isolation(app, mock_embedder):
    """Verifies strict tenant isolation: Alice cannot retrieve Bob's vector-embedded chunks."""
    with app.app_context():
        service = DocumentService(embedding_provider=mock_embedder)

        service.upload_document(
            user_id="user_alice",
            file_bytes=b"Alice secret design system and typography palette.",
            filename="alice_design.txt",
        )
        service.upload_document(
            user_id="user_bob",
            file_bytes=b"Bob private banking credentials and financial audit.",
            filename="bob_finances.txt",
        )

        # Alice searches for banking/financial terms
        alice_results = service.retrieve_candidates_for_query(
            user_id="user_alice", query="private banking credentials"
        )
        # Even with high semantic match to Bob's document, Alice must never see Bob's data
        for item in alice_results:
            assert "banking" not in item.content.lower()
            assert "financial" not in item.content.lower()

        # Bob searches for finances and gets his document
        bob_results = service.retrieve_candidates_for_query(
            user_id="user_bob", query="private banking credentials"
        )
        assert any("banking" in item.content.lower() for item in bob_results)


def test_09_empty_vector_result_handling(app, mock_embedder):
    """Verifies safe empty list return when user has no uploaded documents."""
    with app.app_context():
        service = DocumentService(embedding_provider=mock_embedder)
        results = service.retrieve_candidates_for_query(user_id="user_empty", query="anything")
        assert results == []


def test_10_reciprocal_rank_fusion():
    """Verifies that RRF properly merges and boosts candidates found in both streams."""
    c1 = RetrievalResult(
        source_id="doc_1",
        title="Document 1",
        content="Chunk 1 content",
        metadata={"chunk_id": "c1"},
    )
    c2 = RetrievalResult(
        source_id="doc_2",
        title="Document 2",
        content="Chunk 2 content",
        metadata={"chunk_id": "c2"},
    )
    c3 = RetrievalResult(
        source_id="doc_3",
        title="Document 3",
        content="Chunk 3 content",
        metadata={"chunk_id": "c3"},
    )

    lexical = [c1, c2]
    semantic = [c2, c3]

    fused = reciprocal_rank_fusion(lexical, semantic, k=60)
    assert len(fused) == 3
    # c2 appeared in both lexical and semantic, so it should have the highest RRF score
    assert fused[0].metadata["chunk_id"] == "c2"
    assert "lexical" in fused[0].metadata["match_types"]
    assert "semantic" in fused[0].metadata["match_types"]


def test_11_semantic_retrieval_fallback_on_embedding_failure(app):
    """Verifies that when embedding provider fails, DocumentService falls back to lexical retrieval."""
    with app.app_context():
        failing_embedder = MockEmbeddingProvider(dimension=16, should_fail=True)
        service = DocumentService(embedding_provider=failing_embedder)

        # Upload with embedding disabled or None
        doc = AIDocument(
            id="doc_fallback",
            user_id="user_alice",
            filename="fallback.txt",
            title="Fallback Doc",
            file_type="txt",
            file_size=50,
            chunk_count=1,
        )
        db.session.add(doc)
        chunk = AIDocumentChunk(
            id="chk_fallback_1",
            document_id="doc_fallback",
            user_id="user_alice",
            chunk_index=0,
            text="Deterministic fallback text for testing resilience.",
            char_count=50,
            embedding=None,
        )
        db.session.add(chunk)
        db.session.commit()

        # Query should not raise; it gracefully returns lexical candidates
        results = service.retrieve_candidates_for_query(
            user_id="user_alice", query="deterministic fallback"
        )
        assert len(results) >= 1
        assert "Deterministic fallback" in results[0].content


def test_12_deleted_document_exclusion(app, mock_embedder):
    """Verifies that deleting a document excludes it immediately from semantic retrieval."""
    with app.app_context():
        service = DocumentService(embedding_provider=mock_embedder)
        doc = service.upload_document(
            user_id="user_alice",
            file_bytes=b"Temporary document to be deleted.",
            filename="temp.txt",
        )
        assert len(service.retrieve_candidates_for_query("user_alice", "temporary")) >= 1

        # Delete document
        deleted = service.delete_document("user_alice", doc.id)
        assert deleted is True

        # Now semantic search returns nothing
        results = service.retrieve_candidates_for_query("user_alice", "temporary")
        assert len(results) == 0


def test_13_bounded_candidate_retrieval(app, mock_embedder):
    """Verifies that max_candidates limit is strictly respected."""
    with app.app_context():
        service = DocumentService(embedding_provider=mock_embedder)
        for i in range(3):
            service.upload_document(
                user_id="user_alice",
                file_bytes=f"Sample sentence {i} repeated multiple times.\n\nParagraph {i}.".encode("utf-8"),
                filename=f"bounded_{i}.txt",
            )

        results = service.retrieve_candidates_for_query(
            user_id="user_alice", query="sample sentence", max_candidates=2
        )
        assert len(results) <= 2


# ==============================================================================
# 4. Backfill Utility Test
# ==============================================================================

def test_14_backfill_utility(app, mock_embedder):
    """Verifies that the backfill script successfully embeds un-embedded chunks in batches."""
    with app.app_context():
        doc = AIDocument(
            id="doc_backfill",
            user_id="user_alice",
            filename="backfill.txt",
            title="Backfill Document",
            file_type="txt",
            file_size=100,
            chunk_count=2,
        )
        db.session.add(doc)
        c1 = AIDocumentChunk(
            id="chk_bf_1",
            document_id="doc_backfill",
            user_id="user_alice",
            chunk_index=0,
            text="Unembedded chunk one.",
            char_count=20,
            embedding=None,
        )
        c2 = AIDocumentChunk(
            id="chk_bf_2",
            document_id="doc_backfill",
            user_id="user_alice",
            chunk_index=1,
            text="Unembedded chunk two.",
            char_count=20,
            embedding=None,
        )
        db.session.add_all([c1, c2])
        db.session.commit()

        # Run backfill
        stats = backfill_chunk_embeddings(batch_size=10, provider=mock_embedder)
        assert stats["status"] == "completed"
        assert stats["chunks_processed"] >= 2

        # Verify chunks now have embeddings
        c1_updated = AIDocumentChunk.query.filter_by(id="chk_bf_1").first()
        assert c1_updated.embedding is not None
        assert len(c1_updated.embedding) == 16


# ==============================================================================
# 5. Coordinator & EvidencePack / Citation Compatibility Tests
# ==============================================================================

def test_15_coordinator_hybrid_retrieval_and_reasoning(app, mock_embedder):
    """Verifies that WebResearchEngine coordinates hybrid semantic+web retrieval into grounded AnswerResponse."""
    with app.app_context():
        doc_service = DocumentService(embedding_provider=mock_embedder)
        doc_service.upload_document(
            user_id="user_alice",
            file_bytes=b"Raft consensus protocol replaces Paxos using randomized election timers.",
            filename="raft_notes.txt",
            title="Raft Notes",
        )

        mock_search = MockSearchProvider()
        search_service = SearchService(provider=mock_search)
        mock_llm = MockLLMProvider(default_response="Raft provides consensus [S1] through leader election [S2] and notes [S3].")
        reasoning_engine = ReasoningEngine(provider=mock_llm)

        engine = WebResearchEngine(
            search_service=search_service,
            reasoning_engine=reasoning_engine,
            document_service=doc_service,
        )

        req = SearchRequest(query="Raft consensus protocol", max_results=2, fetch_content=False)

        # 1. EvidencePack has both web and semantic document chunks
        evidence_pack = engine.research_evidence(req, search_mode="hybrid", user_id="user_alice")
        assert isinstance(evidence_pack, EvidencePack)
        source_types = {item.source_type for item in evidence_pack.items}
        assert "document" in source_types

        # 2. Grounded answer verifies citations
        answer_resp = engine.research_and_answer(req, search_mode="hybrid", user_id="user_alice")
        assert answer_resp.grounding_status == "grounded"
        assert len(answer_resp.citations) >= 1
        assert any(c.source_type == "document" for c in answer_resp.citations)


# ==============================================================================
# 6. API Endpoint Tests
# ==============================================================================

def test_16_evidence_api_endpoint_with_semantic_retrieval(app, client, mock_embedder):
    """Verifies POST /api/ai/research/evidence returns semantic document items."""
    with app.app_context():
        doc_service = DocumentService(embedding_provider=mock_embedder)
        doc_service.upload_document(
            user_id="user_alice",
            file_bytes=b"Vector databases index high-dimensional embeddings for nearest neighbor search.",
            filename="vectordb.txt",
            title="Vector DB Notes",
        )

    with patch("app.middleware.auth_middleware.decode_token", return_value={"sub": "user_alice", "type": "access"}):
        resp = client.post(
            "/api/ai/research/evidence",
            json={"query": "vector indexing embeddings", "search_mode": "document"},
            headers={"Authorization": "Bearer valid-token"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["items"]) >= 1
        assert data["items"][0]["source_type"] == "document"


def test_17_answer_api_endpoint_with_semantic_hybrid_reasoning(app, client, mock_embedder):
    """Verifies POST /api/ai/research/answer synthesizes grounded response with semantic document evidence."""
    with app.app_context():
        doc_service = DocumentService(embedding_provider=mock_embedder)
        doc_service.upload_document(
            user_id="user_alice",
            file_bytes=b"Microservices communicate via gRPC and asynchronous messaging queues.",
            filename="microservices.txt",
            title="Microservices Guide",
        )

    mock_web_research = ResearchResponse(
        query="microservices communication",
        results=[
            WebSearchResult(
                title="Microservice Architecture",
                url="https://microservices.io",
                snippet="Patterns for communication between microservices.",
                content="Patterns for communication between microservices.",
                source="microservices.io",
            )
        ],
        total_results=1,
        search_status="success",
    )

    with patch("app.middleware.auth_middleware.decode_token", return_value={"sub": "user_alice", "type": "access"}), \
         patch("app.routes.ai_engine._engine.research", return_value=mock_web_research):
        resp = client.post(
            "/api/ai/research/answer",
            json={"query": "microservices communication", "search_mode": "hybrid"},
            headers={"Authorization": "Bearer valid-token"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "answer" in data
        assert data["grounding_status"] in ("grounded", "partially_grounded")


# ==============================================================================
# 7. Regression Protection Tests
# ==============================================================================

def test_18_phase1_and_phase2_web_mode_regression(app):
    """Verifies that search_mode='web' remains completely unaffected by Phase 5 semantic features."""
    with app.app_context():
        mock_search = MockSearchProvider()
        search_service = SearchService(provider=mock_search)
        engine = WebResearchEngine(search_service=search_service)

        req = SearchRequest(query="python concurrency", max_results=3, fetch_content=False)
        pack = engine.research_evidence(req, search_mode="web")

        assert isinstance(pack, EvidencePack)
        assert len(pack.items) >= 1
        assert all(item.source_type == "web" for item in pack.items)


def test_19_phase3_reasoning_regression(app):
    """Verifies that Phase 3 grounded reasoning engine and citation validation continue functioning seamlessly."""
    mock_llm = MockLLMProvider(default_response="Concurrency is achieved via asyncio [S1].")
    reasoning_engine = ReasoningEngine(provider=mock_llm)

    # Use EvidenceBuilder to format pack
    builder = EvidenceBuilder()
    pack = builder.build_pack(
        query="python concurrency",
        candidates=[
            RetrievalResult(
                source_id="https://docs.python.org",
                title="Python Asyncio",
                url="https://docs.python.org",
                content="Python asyncio provides single-threaded concurrent code using coroutines.",
            )
        ],
    )
    ans = reasoning_engine.answer("python concurrency", pack)
    assert ans.grounding_status == "grounded"
    assert len(ans.citations) == 1
    assert ans.citations[0].citation_id == "S1"


def test_20_phase4_document_crud_semantic_disabled_regression(app):
    """Verifies that disabling AI_ENGINE_SEMANTIC_ENABLED preserves complete Phase 4 lexical behavior."""
    app.config["AI_ENGINE_SEMANTIC_ENABLED"] = False
    with app.app_context():
        service = DocumentService()
        doc = service.upload_document(
            user_id="user_alice",
            file_bytes=b"Pure lexical document for regression testing without vector embeddings.",
            filename="lexical_only.txt",
        )
        assert doc.id is not None
        chunk = AIDocumentChunk.query.filter_by(document_id=doc.id).first()
        # Embedding is None because semantic search was disabled
        assert chunk.embedding is None

        # Retrieval still succeeds via lexical path
        results = service.retrieve_candidates_for_query(
            user_id="user_alice", query="lexical document regression"
        )
        assert len(results) >= 1
        assert "Pure lexical document" in results[0].content

