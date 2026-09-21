"""
tests/test_ai_engine_phase2.py
Comprehensive unit and integration test suite for AI Engine Phase 2 (Retrieval & Evidence Layer).
Covers all 20 required verification scenarios with zero external network access.
"""
import pytest
from unittest.mock import MagicMock, patch

from app import create_app
from app.ai_engine.coordinator import WebResearchEngine
from app.ai_engine.retrieval.chunking import TextChunker
from app.ai_engine.retrieval.evidence import EvidenceBuilder
from app.ai_engine.retrieval.reranker.keyword import KeywordReranker
from app.ai_engine.retrieval.schemas import (
    RetrievalResult,
    EvidenceItem,
    EvidencePack,
    EvidenceRequest,
)
from app.ai_engine.retrieval.service import RetrievalService
from app.ai_engine.schemas.research import (
    ResearchResponse,
    WebSearchResult,
    SourceMetadata,
    SearchRequest,
)


@pytest.fixture
def app():
    """Create Flask test application."""
    test_app = create_app("testing")
    test_app.config["AI_ENGINE_ENABLED"] = True
    test_app.config["RATELIMIT_ENABLED"] = False
    return test_app


@pytest.fixture
def client(app):
    return app.test_client()


# ==============================================================================
# 1. Empty Retrieval Result
# ==============================================================================
def test_01_empty_retrieval_result():
    builder = EvidenceBuilder()
    pack = builder.build_pack(query="python programming", candidates=[])

    assert isinstance(pack, EvidencePack)
    assert pack.query == "python programming"
    assert pack.items == []
    assert pack.total_candidates == 0
    assert pack.selected_items == 0
    assert pack.total_characters == 0
    assert pack.source_summary == []


# ==============================================================================
# 2. Single Result
# ==============================================================================
def test_02_single_result():
    builder = EvidenceBuilder()
    candidate = RetrievalResult(
        source_id="https://example.com/single",
        source_type="web",
        title="Python Guide",
        url="https://example.com/single",
        content="Python is an interpreted, high-level, general-purpose programming language.",
    )

    pack = builder.build_pack(query="python programming language", candidates=[candidate])

    assert pack.total_candidates == 1
    assert pack.selected_items >= 1
    assert pack.items[0].source_id == "https://example.com/single"
    assert pack.items[0].title == "Python Guide"
    assert "Python" in pack.items[0].text
    assert len(pack.source_summary) == 1
    assert pack.source_summary[0].source_id == "https://example.com/single"


# ==============================================================================
# 3. Multiple Results
# ==============================================================================
def test_03_multiple_results():
    builder = EvidenceBuilder()
    candidates = [
        RetrievalResult(
            source_id="https://example.com/doc1",
            title="FastAPI Web Framework",
            url="https://example.com/doc1",
            content="FastAPI is a modern, fast web framework for building APIs with Python 3.8+.",
        ),
        RetrievalResult(
            source_id="https://example.com/doc2",
            title="Flask Microframework",
            url="https://example.com/doc2",
            content="Flask is a lightweight WSGI web application framework in Python.",
        ),
    ]

    pack = builder.build_pack(query="python web framework", candidates=candidates)

    assert pack.total_candidates == 2
    assert pack.selected_items >= 2
    source_ids = {item.source_id for item in pack.items}
    assert "https://example.com/doc1" in source_ids
    assert "https://example.com/doc2" in source_ids
    assert len(pack.source_summary) == 2


# ==============================================================================
# 4. Duplicate URLs
# ==============================================================================
def test_04_duplicate_urls():
    service = RetrievalService()
    candidates = [
        RetrievalResult(
            source_id="https://example.com/page",
            title="Page Version 1",
            url="https://example.com/page",
            content="This is the first version of the document content.",
        ),
        RetrievalResult(
            source_id="https://example.com/page",
            title="Page Version 2",
            url="https://example.com/page",
            content="This is the second version of the document content.",
        ),
    ]

    deduped = service.deduplicate_candidates(candidates)
    assert len(deduped) == 1
    assert deduped[0].url == "https://example.com/page"


# ==============================================================================
# 5. Normalized Duplicate URLs
# ==============================================================================
def test_05_normalized_duplicate_urls():
    service = RetrievalService()
    candidates = [
        RetrievalResult(
            source_id="https://example.com/article/",
            title="Canonical Article",
            url="https://example.com/article/",
            content="Canonical article body text discussing software design.",
        ),
        RetrievalResult(
            source_id="https://example.com/article?utm_source=twitter&utm_medium=social",
            title="Tracked Article",
            url="https://example.com/article?utm_source=twitter&utm_medium=social",
            content="Tracked article body text discussing software design.",
        ),
    ]

    deduped = service.deduplicate_candidates(candidates)
    assert len(deduped) == 1
    # Both normalize to https://example.com/article
    assert deduped[0].title == "Canonical Article"


# ==============================================================================
# 6. Empty Content
# ==============================================================================
def test_06_empty_content():
    service = RetrievalService()
    candidates = [
        RetrievalResult(
            source_id="https://example.com/empty",
            title="Empty Document",
            url="https://example.com/empty",
            content="   ",
            snippet=None,
        ),
        RetrievalResult(
            source_id="https://example.com/valid",
            title="Valid Document",
            url="https://example.com/valid",
            content="Valid content that exceeds minimum required length threshold.",
        ),
    ]

    filtered = service.filter_quality(candidates)
    assert len(filtered) == 1
    assert filtered[0].source_id == "https://example.com/valid"


# ==============================================================================
# 7. Very Large Content
# ==============================================================================
def test_07_very_large_content():
    chunker = TextChunker(max_chunk_chars=300, overlap_chars=30)
    # Generate > 1200 chars of structured text to guarantee 3+ chunks
    large_text = (
        "Paragraph one discusses foundational principles of artificial intelligence and systems design.\n\n"
        "Paragraph two explores retrieval augmented generation architectures and vector similarity algorithms.\n\n"
        "Paragraph three analyzes context budget optimization, token efficiency, and reranker algorithms.\n\n"
        "Paragraph four summarizes deployment architectures, production reliability, and evaluation metrics.\n\n"
        "Paragraph five discusses latency, streaming response pipelines, and continuous integration.\n\n"
        "Paragraph six examines fault tolerance, distributed key-value stores, and consensus protocols.\n\n"
        "Paragraph seven evaluates semantic caching, token limits, and prompt compression techniques.\n\n"
        "Paragraph eight concludes with comprehensive system verification, audit logs, and performance metrics."
    )

    chunks = chunker.chunk_text(large_text)
    assert len(chunks) >= 3
    for ch in chunks:
        assert len(ch.text) <= 300
        assert ch.total_chunks == len(chunks)
        assert ch.chunk_index >= 0


# ==============================================================================
# 8. Chunking Details (Boundary awareness & Overlap)
# ==============================================================================
def test_08_chunking_details():
    chunker = TextChunker(max_chunk_chars=120, overlap_chars=20)
    text = (
        "First sentence in section one. Second sentence in section one.\n\n"
        "Third sentence in section two. Fourth sentence in section two."
    )
    chunks = chunker.chunk_text(text)

    assert len(chunks) >= 2
    # Verify metadata fields
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[0].total_chunks == len(chunks)
    assert chunks[1].total_chunks == len(chunks)
    assert chunks[0].char_count == len(chunks[0].text)


# ==============================================================================
# 9. Query Relevance Ranking
# ==============================================================================
def test_09_query_relevance_ranking():
    reranker = KeywordReranker(min_relevance_score=0.0, filter_low_relevance=False)
    items = [
        EvidenceItem(
            evidence_id="evi_low",
            source_id="src_low",
            title="General Technology",
            text="Today we discuss general trends in cloud computing and serverless architectures.",
        ),
        EvidenceItem(
            evidence_id="evi_high",
            source_id="src_high",
            title="Python Async Mastery",
            text="Python async programming with asyncio allows high concurrency and non-blocking I/O.",
        ),
    ]

    ranked = reranker.rank(query="python async programming", items=items)
    assert ranked[0].evidence_id == "evi_high"
    assert ranked[0].relevance_score > ranked[1].relevance_score


# ==============================================================================
# 10. Title Relevance
# ==============================================================================
def test_10_title_relevance():
    reranker = KeywordReranker(min_relevance_score=0.0, filter_low_relevance=False)
    # Both items have identical text overlap, but item_title has query in title
    item_title = EvidenceItem(
        evidence_id="evi_title",
        source_id="src_title",
        title="Docker Container Security Best Practices",
        text="Security practices involve minimizing attack surfaces and running non-root users.",
    )
    item_notitle = EvidenceItem(
        evidence_id="evi_notitle",
        source_id="src_notitle",
        title="Miscellaneous Notes",
        text="Security practices involve minimizing attack surfaces and running non-root users.",
    )

    ranked = reranker.rank(query="docker container security", items=[item_notitle, item_title])
    assert ranked[0].evidence_id == "evi_title"
    assert ranked[0].relevance_score > ranked[1].relevance_score


# ==============================================================================
# 11. Exact Phrase Relevance
# ==============================================================================
def test_11_exact_phrase_relevance():
    reranker = KeywordReranker(min_relevance_score=0.0, filter_low_relevance=False)
    exact_phrase_item = EvidenceItem(
        evidence_id="evi_exact",
        source_id="src1",
        title="Machine Learning Guide",
        text="In this tutorial we explore deep reinforcement learning from human feedback.",
    )
    scattered_item = EvidenceItem(
        evidence_id="evi_scattered",
        source_id="src2",
        title="Deep Dive into Algorithms",
        text="Learning about human behavior requires reinforcement of deep statistical methods.",
    )

    ranked = reranker.rank(query="deep reinforcement learning", items=[scattered_item, exact_phrase_item])
    assert ranked[0].evidence_id == "evi_exact"
    assert ranked[0].relevance_score > ranked[1].relevance_score


# ==============================================================================
# 12. Low Relevance Result Filtering
# ==============================================================================
def test_12_low_relevance_filtering():
    reranker = KeywordReranker(min_relevance_score=0.20, filter_low_relevance=True)
    items = [
        EvidenceItem(
            evidence_id="evi_relevant",
            source_id="src_rel",
            title="Kubernetes Ingress",
            text="Kubernetes ingress controllers manage external HTTP and HTTPS traffic routing.",
        ),
        EvidenceItem(
            evidence_id="evi_irrelevant",
            source_id="src_irrel",
            title="Baking Sourdough",
            text="Fermenting sourdough bread requires water, flour, wild yeast, and salt.",
        ),
    ]

    ranked = reranker.rank(query="kubernetes ingress routing", items=items)
    evidence_ids = [item.evidence_id for item in ranked]
    assert "evi_relevant" in evidence_ids
    assert "evi_irrelevant" not in evidence_ids


# ==============================================================================
# 13. Stable Ranking
# ==============================================================================
def test_13_stable_ranking():
    reranker = KeywordReranker(min_relevance_score=0.0, filter_low_relevance=False)
    # Two identical items with same title and text but distinct deterministic IDs
    items = [
        EvidenceItem(
            evidence_id="evi_b",
            source_id="src_b",
            title="Same Title",
            text="Identical text body describing identical concepts and tokens.",
        ),
        EvidenceItem(
            evidence_id="evi_a",
            source_id="src_a",
            title="Same Title",
            text="Identical text body describing identical concepts and tokens.",
        ),
    ]

    run_1 = reranker.rank(query="identical concepts", items=list(items))
    run_2 = reranker.rank(query="identical concepts", items=list(reversed(items)))

    # Order must be strictly identical across independent runs regardless of input order
    assert [x.evidence_id for x in run_1] == [x.evidence_id for x in run_2]


# ==============================================================================
# 14. Evidence Provenance
# ==============================================================================
def test_14_evidence_provenance():
    builder = EvidenceBuilder()
    candidates = [
        RetrievalResult(
            source_id="https://nirvexa.in/architecture",
            source_type="web",
            title="Nirvexa Architecture",
            url="https://nirvexa.in/architecture",
            content="The Nirvexa platform integrates advanced agentic coding workflows with high precision.",
        )
    ]

    pack = builder.build_pack(query="nirvexa platform architecture", candidates=candidates)

    for item in pack.items:
        # Every item must have valid source_id, title, url, and source_type
        assert item.source_id == "https://nirvexa.in/architecture"
        assert item.url == "https://nirvexa.in/architecture"
        assert item.title == "Nirvexa Architecture"
        assert item.source_type == "web"
        assert item.evidence_id.startswith("evi_")
        assert "chunk_index" in item.metadata


# ==============================================================================
# 15. Citation Metadata Preservation
# ==============================================================================
def test_15_citation_metadata_preservation():
    builder = EvidenceBuilder()
    candidates = [
        RetrievalResult(
            source_id="https://source-a.org/research",
            source_type="web",
            title="Research Source A",
            url="https://source-a.org/research",
            content="Research findings on distributed systems consensus protocols.\n\nAdditional consensus analysis.",
        ),
        RetrievalResult(
            source_id="doc_internal_123",
            source_type="document",
            title="Internal Whitepaper",
            url=None,
            content="Internal whitepaper on distributed fault tolerance.",
        ),
    ]

    pack = builder.build_pack(query="distributed consensus fault tolerance", candidates=candidates)

    # Source summaries must match items selected
    summary_map = {s.source_id: s for s in pack.source_summary}
    assert "https://source-a.org/research" in summary_map
    assert "doc_internal_123" in summary_map

    for item in pack.items:
        assert item.source_id in summary_map
        summary = summary_map[item.source_id]
        assert summary.title == item.title
        assert summary.source_type == item.source_type
        assert summary.chunk_count >= 1


# ==============================================================================
# 16. Maximum Evidence Item Limit
# ==============================================================================
def test_16_max_evidence_item_limit():
    builder = EvidenceBuilder()
    # Provide 5 candidate documents with relevant content
    candidates = [
        RetrievalResult(
            source_id=f"https://example.com/doc_{i}",
            title=f"Cloud Guide {i}",
            content=f"Cloud computing infrastructure and distributed storage {i}.",
        )
        for i in range(5)
    ]

    pack = builder.build_pack(
        query="cloud computing",
        candidates=candidates,
        max_evidence_items=2,  # Strict cap
    )

    assert len(pack.items) == 2
    assert pack.selected_items == 2


# ==============================================================================
# 17. Maximum Evidence Character Limit
# ==============================================================================
def test_17_max_evidence_character_limit():
    builder = EvidenceBuilder()
    candidates = [
        RetrievalResult(
            source_id="https://example.com/long",
            title="Long Overview",
            content=(
                "First section on algorithms and computation.\n\n"
                "Second section on memory architectures and cache coherency.\n\n"
                "Third section on processor execution pipelines."
            ),
        )
    ]

    pack = builder.build_pack(
        query="algorithms architectures",
        candidates=candidates,
        max_evidence_chars=120,  # Cap at 120 chars
    )

    assert pack.total_characters <= 120
    for item in pack.items:
        assert len(item.text) <= 120


# ==============================================================================
# 18. Malformed Retrieval Result
# ==============================================================================
def test_18_malformed_retrieval_result():
    service = RetrievalService()
    # Candidate with missing/None fields
    malformed = [
        {"title": None, "url": None, "content": "Valid text inside a dictionary candidate with null fields."},
        {"invalid_key": 123},  # completely missing content
        None,  # None item
    ]

    normalized = service.from_raw_candidates(malformed)  # type: ignore
    assert len(normalized) == 2
    valid = service.filter_quality(normalized)
    assert len(valid) == 1
    assert "Valid text" in valid[0].content


# ==============================================================================
# 19. Missing Metadata
# ==============================================================================
def test_19_missing_metadata():
    builder = EvidenceBuilder()
    candidate = RetrievalResult(
        source_id="https://example.com/no-meta",
        title="No Metadata Page",
        url="https://example.com/no-meta",
        content="Content from a source that lacks any HTML meta tags or attributes.",
        metadata={},  # Empty metadata
    )

    pack = builder.build_pack(query="content without metadata", candidates=[candidate])
    assert pack.selected_items >= 1
    item = pack.items[0]
    assert isinstance(item.metadata, dict)
    assert "chunk_index" in item.metadata
    assert item.metadata["source_id"] == "https://example.com/no-meta"


# ==============================================================================
# 20. Phase 1 Regression & Coordinator Integration
# ==============================================================================
def test_20_coordinator_evidence_integration_mock():
    """Verify WebResearchEngine.research_evidence seamlessly builds an EvidencePack."""
    from app.ai_engine.search.provider import MockSearchProvider
    from app.ai_engine.search.service import SearchService

    mock_provider = MockSearchProvider()
    search_service = SearchService(provider=mock_provider)
    engine = WebResearchEngine(search_service=search_service)

    req = SearchRequest(query="machine learning engineering", max_results=3, fetch_content=False)

    # Test Phase 1 still works
    phase1_resp = engine.research(req)
    assert isinstance(phase1_resp, ResearchResponse)
    assert len(phase1_resp.results) > 0

    # Test Phase 2 builds evidence
    pack = engine.research_evidence(req, max_evidence_items=3)
    assert isinstance(pack, EvidencePack)
    assert pack.query == "machine learning engineering"
    assert pack.selected_items > 0
    assert len(pack.items) <= 3


# ==============================================================================
# API Endpoint Tests for POST /api/ai/research/evidence
# ==============================================================================
class TestEvidenceEndpoint:

    def test_evidence_endpoint_unauthorized_without_token(self, app):
        with app.test_client() as client:
            resp = client.post(
                "/api/ai/research/evidence",
                json={"query": "python async"},
            )
            assert resp.status_code == 401

    def test_evidence_endpoint_feature_flag_disabled(self, app):
        with app.test_client() as client:
            app.config["AI_ENGINE_ENABLED"] = False
            with patch("app.middleware.auth_middleware.decode_token", return_value={"sub": "user1", "type": "access"}):
                resp = client.post(
                    "/api/ai/research/evidence",
                    json={"query": "python async"},
                    headers={"Authorization": "Bearer valid-token"},
                )
                assert resp.status_code == 503
                data = resp.get_json()
                assert data["error"] == "ai_engine_disabled"

    def test_evidence_endpoint_validation_error_empty_query(self, app):
        with app.test_client() as client:
            app.config["AI_ENGINE_ENABLED"] = True
            with patch("app.middleware.auth_middleware.decode_token", return_value={"sub": "user1", "type": "access"}):
                resp = client.post(
                    "/api/ai/research/evidence",
                    json={"query": ""},
                    headers={"Authorization": "Bearer valid-token"},
                )
                assert resp.status_code == 400
                data = resp.get_json()
                assert data["error"] == "validation_error"

    def test_evidence_endpoint_success(self, app):
        with app.test_client() as client:
            app.config["AI_ENGINE_ENABLED"] = True
            mock_pack = EvidencePack(
                query="fastapi performance",
                items=[
                    EvidenceItem(
                        evidence_id="evi_test_1",
                        source_id="https://fastapi.tiangolo.com",
                        text="FastAPI provides very high performance on par with NodeJS and Go.",
                        title="FastAPI Benchmarks",
                        url="https://fastapi.tiangolo.com",
                        relevance_score=0.88,
                    )
                ],
                total_candidates=1,
                selected_items=1,
                total_characters=68,
            )

            with patch("app.middleware.auth_middleware.decode_token", return_value={"sub": "user1", "type": "access"}), \
                 patch("app.routes.ai_engine._engine.research_evidence", return_value=mock_pack):
                resp = client.post(
                    "/api/ai/research/evidence",
                    json={"query": "fastapi performance", "max_evidence_items": 3},
                    headers={"Authorization": "Bearer valid-token"},
                )
                assert resp.status_code == 200
                data = resp.get_json()
                assert data["query"] == "fastapi performance"
                assert len(data["items"]) == 1
                assert data["items"][0]["evidence_id"] == "evi_test_1"
                assert data["items"][0]["relevance_score"] == 0.88
