"""
tests/test_ai_engine_phase6.py
Unit, integration, fallback, and regression tests for Phase 6 Step 2:
MultiSignalReranker, RerankerFactory, Multi-Signal Scoring, and EvidenceBuilder Integration.
Zero live external network calls.
"""
from datetime import datetime, timezone, timedelta
import pytest
from unittest.mock import patch, MagicMock

from app import create_app
from app.extensions import db
from app.models.ai_document import AIDocument, AIDocumentChunk
from app.ai_engine.coordinator import WebResearchEngine
from app.ai_engine.documents.service import DocumentService
from app.ai_engine.embeddings.mock import MockEmbeddingProvider
from app.ai_engine.reasoning.engine import ReasoningEngine
from app.ai_engine.reasoning.providers.mock import MockLLMProvider
from app.ai_engine.retrieval.evidence import EvidenceBuilder
from app.ai_engine.retrieval.reranker.base import BaseReranker
from app.ai_engine.retrieval.reranker.keyword import KeywordReranker
from app.ai_engine.retrieval.reranker.multi_signal import MultiSignalReranker
from app.ai_engine.retrieval.reranker.factory import RerankerFactory
from app.ai_engine.retrieval.schemas import EvidenceItem, RetrievalResult, EvidencePack
from app.ai_engine.schemas.research import SearchRequest, ResearchResponse, WebSearchResult
from app.ai_engine.search.provider import MockSearchProvider
from app.ai_engine.search.service import SearchService


@pytest.fixture
def app():
    """Create Flask test application configured for Phase 6 testing with in-memory DB."""
    test_app = create_app("testing")
    test_app.config["AI_ENGINE_ENABLED"] = True
    test_app.config["AI_ENGINE_LLM_ENABLED"] = True
    test_app.config["AI_ENGINE_DOCUMENTS_ENABLED"] = True
    test_app.config["AI_ENGINE_SEMANTIC_ENABLED"] = True
    test_app.config["AI_ENGINE_EMBEDDING_PROVIDER"] = "mock"
    test_app.config["AI_ENGINE_EMBEDDING_DIMENSION"] = 16
    test_app.config["AI_ENGINE_RERANKER_TYPE"] = "keyword"
    test_app.config["AI_ENGINE_LLM_PROVIDER"] = "mock"
    test_app.config["RATELIMIT_ENABLED"] = False

    with test_app.app_context():
        AIDocument.__table__.create(db.engine, checkfirst=True)
        AIDocumentChunk.__table__.create(db.engine, checkfirst=True)
        yield test_app
        db.session.remove()
        AIDocumentChunk.__table__.drop(db.engine, checkfirst=True)
        AIDocument.__table__.drop(db.engine, checkfirst=True)


@pytest.fixture
def mock_embedder():
    return MockEmbeddingProvider(dimension=16)


def _make_item(
    evidence_id: str,
    text: str,
    title: str = "Sample Title",
    source_id: str = "src_1",
    source_type: str = "web",
    url: str = "https://example.com/doc",
    metadata: dict = None,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        source_id=source_id,
        text=text,
        title=title,
        url=url,
        source_type=source_type,
        relevance_score=0.0,
        metadata=metadata or {},
    )


# ==============================================================================
# 1. Initialization and Weight Configuration
# ==============================================================================

def test_01_multisignal_reranker_initialization_defaults(app):
    """Verifies default weight loading and initialization."""
    with app.app_context():
        reranker = MultiSignalReranker()
        assert reranker.weight_lexical == 0.35
        assert reranker.weight_semantic == 0.30
        assert reranker.weight_source_quality == 0.15
        assert reranker.weight_freshness == 0.10
        assert reranker.weight_diversity == 0.10
        assert reranker.min_relevance_score == 0.10


def test_02_weight_clamping_and_extreme_values(app):
    """Verifies that weights are safely clamped to their defined ranges."""
    reranker = MultiSignalReranker(
        weight_lexical=-0.5,
        weight_semantic=2.5,
        weight_source_quality=1.5,
        weight_freshness=-1.0,
        weight_diversity=0.99,
        min_relevance_score=-0.2,
    )
    assert reranker.weight_lexical == 0.0
    assert reranker.weight_semantic == 1.0
    assert reranker.weight_source_quality == 1.0
    assert reranker.weight_freshness == 0.0
    assert reranker.weight_diversity == 0.50  # clamped to max 0.50
    assert reranker.min_relevance_score == 0.0


def test_03_factory_resolution(app):
    """Verifies RerankerFactory resolves keyword vs multi_signal properly."""
    with app.app_context():
        app.config["AI_ENGINE_RERANKER_TYPE"] = "keyword"
        r_key = RerankerFactory.get_reranker()
        assert isinstance(r_key, KeywordReranker)

        app.config["AI_ENGINE_RERANKER_TYPE"] = "multi_signal"
        r_multi = RerankerFactory.get_reranker()
        assert isinstance(r_multi, MultiSignalReranker)

        # Invalid type safely resolves to KeywordReranker
        app.config["AI_ENGINE_RERANKER_TYPE"] = "cohere_unsupported"
        r_invalid = RerankerFactory.get_reranker()
        assert isinstance(r_invalid, KeywordReranker)


# ==============================================================================
# 2. Individual Signal Verification
# ==============================================================================

def test_04_lexical_signal_dominance_when_other_weights_zero():
    """Verifies lexical scoring when weight_lexical is 1.0 and others are 0.0."""
    reranker = MultiSignalReranker(
        weight_lexical=1.0,
        weight_semantic=0.0,
        weight_source_quality=0.0,
        weight_freshness=0.0,
        weight_diversity=0.0,
        filter_low_relevance=False,
    )
    item_relevant = _make_item("e1", "Kubernetes cluster orchestrates container workloads efficiently.")
    item_irrelevant = _make_item("e2", "French pastry baking requires flour, butter, and patience.")

    ranked = reranker.rank("kubernetes container workloads", [item_irrelevant, item_relevant])
    assert len(ranked) == 2
    assert ranked[0].evidence_id == "e1"
    assert ranked[0].relevance_score > ranked[1].relevance_score


def test_05_semantic_signal_boosts_synonym_match():
    """Verifies that high semantic similarity (low cosine_distance) boosts candidate lacking exact keywords."""
    reranker = MultiSignalReranker(
        weight_lexical=0.20,
        weight_semantic=0.80,
        weight_source_quality=0.0,
        weight_freshness=0.0,
        weight_diversity=0.0,
        filter_low_relevance=False,
    )
    # item_synonym has low exact lexical match but high semantic score (distance 0.05 -> sim 0.95)
    item_synonym = _make_item(
        "e_syn",
        "Container scheduling engine distributes microservices across bare-metal hardware nodes.",
        metadata={"cosine_distance": 0.05},
    )
    # item_lexical has exact token 'kubernetes' but low semantic similarity (distance 0.90 -> sim 0.10)
    item_lexical = _make_item(
        "e_lex",
        "Kubernetes mention without deep semantic context.",
        metadata={"cosine_distance": 0.90},
    )

    ranked = reranker.rank("kubernetes orchestration", [item_lexical, item_synonym])
    assert ranked[0].evidence_id == "e_syn"
    assert "rerank_signals" in ranked[0].metadata
    assert ranked[0].metadata["rerank_signals"]["semantic"] == 0.95


def test_06_source_quality_prioritizes_private_documents_and_authoritative_tlds():
    """Verifies source quality scores private documents highest, followed by authoritative domains."""
    reranker = MultiSignalReranker(
        weight_lexical=0.0,
        weight_semantic=0.0,
        weight_source_quality=1.0,
        weight_freshness=0.0,
        weight_diversity=0.0,
        filter_low_relevance=False,
    )
    doc_item = _make_item("e_doc", "Internal architecture documentation.", source_type="document")
    gov_item = _make_item("e_gov", "NIST cybersecurity standards.", source_type="web", url="https://csrc.nist.gov/pubs")
    standard_web = _make_item("e_web", "Blog post on security.", source_type="web", url="https://random-blog.xyz/post")

    ranked = reranker.rank("cybersecurity standards", [standard_web, gov_item, doc_item])
    # Order should be: private document (1.0), gov (0.85-0.90), standard web (0.70)
    assert ranked[0].evidence_id == "e_doc"
    assert ranked[0].metadata["rerank_signals"]["source_quality"] == 1.0
    assert ranked[1].evidence_id == "e_gov"
    assert ranked[1].metadata["rerank_signals"]["source_quality"] >= 0.85
    assert ranked[2].evidence_id == "e_web"


def test_07_freshness_signal_prefers_recent_content():
    """Verifies that fresh content receives higher scores than dated content."""
    reranker = MultiSignalReranker(
        weight_lexical=0.0,
        weight_semantic=0.0,
        weight_source_quality=0.0,
        weight_freshness=1.0,
        weight_diversity=0.0,
        filter_low_relevance=False,
    )
    now = datetime.now(timezone.utc)
    recent_ts = (now - timedelta(days=2)).isoformat()
    old_ts = (now - timedelta(days=800)).isoformat()

    recent_item = _make_item("e_recent", "AI developments in 2026.", metadata={"published_at": recent_ts})
    old_item = _make_item("e_old", "AI developments in 2023.", metadata={"published_at": old_ts})

    ranked = reranker.rank("AI developments", [old_item, recent_item])
    assert ranked[0].evidence_id == "e_recent"
    assert ranked[0].metadata["rerank_signals"]["freshness"] > ranked[1].metadata["rerank_signals"]["freshness"]


# ==============================================================================
# 3. Diversity and Redundancy Suppression
# ==============================================================================

def test_08_diversity_penalty_penalizes_duplicate_sources_and_repetitive_text():
    """Verifies that items from the same source and near-identical texts are penalized by diversity."""
    reranker = MultiSignalReranker(
        weight_lexical=0.50,
        weight_semantic=0.0,
        weight_source_quality=0.0,
        weight_freshness=0.0,
        weight_diversity=0.30,  # strong diversity penalty
        filter_low_relevance=False,
    )
    # Source A has two nearly identical chunks with exact phrase
    item_a1 = _make_item(
        "e_a1",
        "PostgreSQL vector search provides similarity indexing with HNSW and IVFFlat.",
        source_id="source_A",
    )
    item_a2 = _make_item(
        "e_a2",
        "PostgreSQL vector search provides similarity indexing with HNSW and IVFFlat indexes.",
        source_id="source_A",
    )
    # Source B has complementary different text with exact phrase
    item_b = _make_item(
        "e_b",
        "PostgreSQL vector search enables dense embedding queries in relational databases.",
        source_id="source_B",
    )

    # One item from source_A and one from source_B are selected in top 2;
    # redundant duplicate item_a2 from source_A is penalized and placed last
    ranked = reranker.rank("PostgreSQL vector search", [item_a1, item_a2, item_b])
    top_two = {ranked[0].evidence_id, ranked[1].evidence_id}
    assert top_two == {"e_a1", "e_b"}
    assert ranked[2].evidence_id == "e_a2"
    assert ranked[2].metadata["rerank_signals"]["diversity_penalty"] > 0.15


# ==============================================================================
# 4. Resilience, Missing Metadata, and Edge Cases
# ==============================================================================

def test_09_empty_and_single_candidate_handling():
    """Verifies empty list and single candidate edge cases."""
    reranker = MultiSignalReranker()
    assert reranker.rank("query", []) == []

    single = _make_item("e1", "Single candidate text for validation.")
    res = reranker.rank("validation", [single])
    assert len(res) == 1
    assert res[0].evidence_id == "e1"
    assert res[0].relevance_score > 0.0


def test_10_missing_and_malformed_metadata_gracefully_handled():
    """Verifies that missing timestamps, unparseable dates, and null metadata do not fail."""
    reranker = MultiSignalReranker(filter_low_relevance=False)
    items = [
        _make_item("e1", "Content without metadata dictionary.", metadata=None),
        _make_item("e2", "Content with unparseable timestamp.", metadata={"published_at": "not-a-date"}),
        _make_item("e3", "Content with invalid cosine distance.", metadata={"cosine_distance": "invalid"}),
    ]
    ranked = reranker.rank("content validation", items)
    assert len(ranked) == 3
    # Neutral fallbacks applied safely
    for it in ranked:
        assert 0.0 <= it.relevance_score <= 1.0
        assert "rerank_signals" in it.metadata


def test_11_deterministic_ranking_guarantee():
    """Verifies that identical query and candidates produce exact identical scores and order."""
    reranker = MultiSignalReranker()
    items = [
        _make_item("e1", "Distributed consensus with Raft leader election."),
        _make_item("e2", "Paxos consensus protocol algorithm."),
        _make_item("e3", "Vector embeddings in nearest neighbor search."),
    ]
    run1 = reranker.rank("consensus protocol", items)
    run2 = reranker.rank("consensus protocol", items)

    assert [x.evidence_id for x in run1] == [x.evidence_id for x in run2]
    assert [x.relevance_score for x in run1] == [x.relevance_score for x in run2]


def test_12_graceful_fallback_to_keyword_on_exception():
    """Verifies that if multi-signal scoring raises an unhandled error, it falls back to KeywordReranker."""
    reranker = MultiSignalReranker()
    item = _make_item("e1", "Important text.")

    with patch.object(reranker, "_rank_multi_signal", side_effect=RuntimeError("Simulated failure")):
        ranked = reranker.rank("important", [item])
        # Did not raise RuntimeError; safely fell back
        assert len(ranked) == 1
        assert ranked[0].evidence_id == "e1"


# ==============================================================================
# 5. EvidenceBuilder Integration
# ==============================================================================

def test_13_evidence_builder_uses_configured_reranker(app):
    """Verifies EvidenceBuilder dynamically picks reranker based on AI_ENGINE_RERANKER_TYPE."""
    with app.app_context():
        # 1. Default config uses KeywordReranker
        app.config["AI_ENGINE_RERANKER_TYPE"] = "keyword"
        builder_kw = EvidenceBuilder()
        assert isinstance(builder_kw.reranker, KeywordReranker)

        # 2. Configured for multi_signal
        app.config["AI_ENGINE_RERANKER_TYPE"] = "multi_signal"
        builder_ms = EvidenceBuilder()
        assert isinstance(builder_ms.reranker, MultiSignalReranker)


def test_14_evidence_builder_build_pack_with_multisignal(app):
    """Verifies that EvidenceBuilder creates valid EvidencePack with explainability metadata."""
    app.config["AI_ENGINE_RERANKER_TYPE"] = "multi_signal"
    with app.app_context():
        builder = EvidenceBuilder()
        candidates = [
            RetrievalResult(
                source_id="https://kubernetes.io/docs",
                title="Kubernetes Architecture",
                url="https://kubernetes.io/docs",
                content="Kubernetes is an open-source container orchestration system for automating application deployment.",
                metadata={"domain": "kubernetes.io", "published_at": "2026-01-01T00:00:00Z"},
            ),
            RetrievalResult(
                source_id="https://random-notes.org",
                title="Random Notes",
                url="https://random-notes.org",
                content="Just some personal notes on baking bread.",
            ),
        ]
        pack = builder.build_pack(query="kubernetes deployment", candidates=candidates)

        assert isinstance(pack, EvidencePack)
        assert len(pack.items) >= 1
        top_item = pack.items[0]
        assert "kubernetes.io" in top_item.url
        assert "rerank_signals" in top_item.metadata
        assert top_item.relevance_score > 0.10
        assert len(pack.source_summary) >= 1


# ==============================================================================
# 6. KeywordReranker Regression Baseline
# ==============================================================================

def test_15_keyword_reranker_regression_untouched():
    """Verifies KeywordReranker remains 100% functional and backward compatible."""
    kw = KeywordReranker(min_relevance_score=0.10)
    items = [
        _make_item("e1", "Exact phrase match in python asyncio concurrency.", title="Python Concurrency"),
        _make_item("e2", "Unrelated cooking recipe.", title="Cooking"),
    ]
    ranked = kw.rank("python asyncio", items)
    assert len(ranked) == 1
    assert ranked[0].evidence_id == "e1"
    assert ranked[0].relevance_score >= 0.25


# ==============================================================================
# 7. Step 3 End-to-End Pipeline Integration Tests
# ==============================================================================

def test_16_coordinator_web_mode_keyword_vs_multisignal(app):
    """Verifies web-only retrieval works under both keyword and multi_signal rerankers (Requirements A, B)."""
    with app.app_context():
        mock_search = MockSearchProvider()
        search_service = SearchService(provider=mock_search)
        engine = WebResearchEngine(search_service=search_service)
        req = SearchRequest(query="FastAPI web framework", max_results=3, fetch_content=False)

        # 1. Under KeywordReranker
        app.config["AI_ENGINE_RERANKER_TYPE"] = "keyword"
        pack_kw = engine.research_evidence(req, search_mode="web")
        assert isinstance(pack_kw, EvidencePack)
        assert len(pack_kw.items) >= 1
        assert all(item.source_type == "web" for item in pack_kw.items)
        assert "rerank_signals" not in pack_kw.items[0].metadata

        # 2. Under MultiSignalReranker
        app.config["AI_ENGINE_RERANKER_TYPE"] = "multi_signal"
        pack_ms = engine.research_evidence(req, search_mode="web")
        assert isinstance(pack_ms, EvidencePack)
        assert len(pack_ms.items) >= 1
        assert all(item.source_type == "web" for item in pack_ms.items)
        assert "rerank_signals" in pack_ms.items[0].metadata
        assert "lexical" in pack_ms.items[0].metadata["rerank_signals"]


def test_17_coordinator_document_mode_keyword_vs_multisignal(app, mock_embedder):
    """Verifies document-only retrieval works under both keyword and multi_signal rerankers (Requirements C, D)."""
    with app.app_context():
        doc_service = DocumentService(embedding_provider=mock_embedder)
        doc_service.upload_document(
            user_id="user_alice",
            file_bytes=b"PostgreSQL pgvector extension enables efficient vector similarity indexing.",
            filename="pgvector_guide.txt",
            title="pgvector Guide",
        )
        engine = WebResearchEngine(document_service=doc_service)
        req = SearchRequest(query="pgvector similarity indexing", max_results=2, fetch_content=False)

        # 1. Under KeywordReranker
        app.config["AI_ENGINE_RERANKER_TYPE"] = "keyword"
        pack_kw = engine.research_evidence(req, search_mode="document", user_id="user_alice")
        assert isinstance(pack_kw, EvidencePack)
        assert len(pack_kw.items) >= 1
        assert all(item.source_type == "document" for item in pack_kw.items)

        # 2. Under MultiSignalReranker
        app.config["AI_ENGINE_RERANKER_TYPE"] = "multi_signal"
        pack_ms = engine.research_evidence(req, search_mode="document", user_id="user_alice")
        assert isinstance(pack_ms, EvidencePack)
        assert len(pack_ms.items) >= 1
        assert all(item.source_type == "document" for item in pack_ms.items)
        assert "rerank_signals" in pack_ms.items[0].metadata
        assert pack_ms.items[0].metadata["rerank_signals"]["source_quality"] == 1.0


def test_18_coordinator_hybrid_mode_rrf_and_multisignal(app, mock_embedder):
    """Verifies hybrid retrieval fuses web + document candidates and applies multi_signal reranker (Requirements E, F)."""
    with app.app_context():
        doc_service = DocumentService(embedding_provider=mock_embedder)
        doc_service.upload_document(
            user_id="user_alice",
            file_bytes=b"Raft consensus protocol replaces Paxos with understandable leader election.",
            filename="raft_paper.txt",
            title="Raft Paper",
        )
        mock_search = MockSearchProvider()
        search_service = SearchService(provider=mock_search)
        engine = WebResearchEngine(search_service=search_service, document_service=doc_service)

        req = SearchRequest(query="Raft consensus protocol", max_results=2, fetch_content=False)

        # Hybrid under MultiSignalReranker
        app.config["AI_ENGINE_RERANKER_TYPE"] = "multi_signal"
        pack = engine.research_evidence(req, search_mode="hybrid", user_id="user_alice")
        assert isinstance(pack, EvidencePack)
        source_types = {item.source_type for item in pack.items}
        assert "document" in source_types
        # Rerank signals preserved
        for item in pack.items:
            assert "rerank_signals" in item.metadata
            signals = item.metadata["rerank_signals"]
            assert 0.0 <= signals["final_score"] <= 1.0


def test_19_end_to_end_research_and_answer_with_multisignal(app, mock_embedder):
    """Verifies end-to-end grounded answer generation and citation validation under MultiSignalReranker."""
    with app.app_context():
        doc_service = DocumentService(embedding_provider=mock_embedder)
        doc_service.upload_document(
            user_id="user_alice",
            file_bytes=b"Kafka handles distributed event streaming through partitioned log topics.",
            filename="kafka_arch.txt",
            title="Kafka Architecture",
        )
        mock_search = MockSearchProvider()
        search_service = SearchService(provider=mock_search)
        mock_llm = MockLLMProvider(default_response="Kafka streams events via log partitions [S1] and architecture docs [S3].")
        reasoning_engine = ReasoningEngine(provider=mock_llm)

        engine = WebResearchEngine(
            search_service=search_service,
            reasoning_engine=reasoning_engine,
            document_service=doc_service,
        )

        app.config["AI_ENGINE_RERANKER_TYPE"] = "multi_signal"
        req = SearchRequest(query="Kafka event streaming", max_results=2, fetch_content=False)
        answer_resp = engine.research_and_answer(req, search_mode="hybrid", user_id="user_alice")

        assert answer_resp.grounding_status == "grounded"
        assert len(answer_resp.citations) >= 1
        assert any(c.source_type == "document" for c in answer_resp.citations)


def test_20_multitenant_isolation_under_multisignal_reranker(app, mock_embedder):
    """Verifies MultiSignalReranker strictly respects user_id isolation (Requirement N)."""
    with app.app_context():
        doc_service = DocumentService(embedding_provider=mock_embedder)
        # Alice document
        doc_service.upload_document(
            user_id="user_alice",
            file_bytes=b"Alice secret design specification for project Alpha.",
            filename="alice_alpha.txt",
            title="Alice Alpha",
        )
        # Bob document
        doc_service.upload_document(
            user_id="user_bob",
            file_bytes=b"Bob confidential design specification for project Beta.",
            filename="bob_beta.txt",
            title="Bob Beta",
        )

        engine = WebResearchEngine(document_service=doc_service)
        app.config["AI_ENGINE_RERANKER_TYPE"] = "multi_signal"
        req = SearchRequest(query="design specification", max_results=5, fetch_content=False)

        # Alice's search
        pack_alice = engine.research_evidence(req, search_mode="document", user_id="user_alice")
        alice_texts = [item.text for item in pack_alice.items]
        assert any("Alice secret" in t for t in alice_texts)
        assert not any("Bob confidential" in t for t in alice_texts)

        # Bob's search
        pack_bob = engine.research_evidence(req, search_mode="document", user_id="user_bob")
        bob_texts = [item.text for item in pack_bob.items]
        assert any("Bob confidential" in t for t in bob_texts)
        assert not any("Alice secret" in t for t in bob_texts)


def test_21_context_budget_enforcement_with_multisignal(app):
    """Verifies max_evidence_items and max_evidence_chars are strictly respected (Requirement L)."""
    app.config["AI_ENGINE_RERANKER_TYPE"] = "multi_signal"
    with app.app_context():
        builder = EvidenceBuilder()
        candidates = [
            RetrievalResult(
                source_id=f"https://example.com/doc_{i}",
                title=f"Doc {i}",
                url=f"https://example.com/doc_{i}",
                content=f"Important candidate text content for item number {i} with detailed information.",
            )
            for i in range(15)
        ]
        pack = builder.build_pack(
            query="important candidate text",
            candidates=candidates,
            max_evidence_items=3,
            max_evidence_chars=200,
        )
        assert isinstance(pack, EvidencePack)
        assert len(pack.items) <= 3
        assert pack.total_characters <= 200


def test_22_invalid_reranker_config_falls_back_in_coordinator(app):
    """Verifies invalid AI_ENGINE_RERANKER_TYPE safely falls back to KeywordReranker (Requirement G)."""
    app.config["AI_ENGINE_RERANKER_TYPE"] = "cohere_invalid"
    with app.app_context():
        mock_search = MockSearchProvider()
        search_service = SearchService(provider=mock_search)
        engine = WebResearchEngine(search_service=search_service)
        req = SearchRequest(query="testing fallback", max_results=2, fetch_content=False)

        pack = engine.research_evidence(req, search_mode="web")
        assert isinstance(pack, EvidencePack)
        assert len(pack.items) >= 1
        # Resolved to KeywordReranker -> no rerank_signals
        assert "rerank_signals" not in pack.items[0].metadata


def test_23_multisignal_reranker_exception_fallback_in_coordinator(app):
    """Verifies that an exception during multi-signal ranking falls back gracefully (Requirement H)."""
    app.config["AI_ENGINE_RERANKER_TYPE"] = "multi_signal"
    with app.app_context():
        mock_search = MockSearchProvider()
        search_service = SearchService(provider=mock_search)
        engine = WebResearchEngine(search_service=search_service)
        req = SearchRequest(query="testing resilience", max_results=2, fetch_content=False)

        # Inject failure into _rank_multi_signal
        with patch("app.ai_engine.retrieval.reranker.multi_signal.MultiSignalReranker._rank_multi_signal",
                   side_effect=RuntimeError("Simulated multi-signal crash")):
            pack = engine.research_evidence(req, search_mode="web")
            # Did not raise 500 error; successfully built EvidencePack via fallback
            assert isinstance(pack, EvidencePack)
            assert len(pack.items) >= 1

