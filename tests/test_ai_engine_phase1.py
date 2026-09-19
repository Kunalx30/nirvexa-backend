"""
tests/test_ai_engine_phase1.py
Comprehensive unit test suite for Nirvexa AI Engine Phase 1 (Web Research Engine).
100% offline / mocked — zero external internet dependencies.
"""
import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO
import requests

from app import create_app
from app.ai_engine.extraction.content_extractor import ContentExtractor
from app.ai_engine.fetch.url_validator import (
    validate_url,
    normalize_url,
    deduplicate_urls,
)
from app.ai_engine.fetch.web_fetcher import WebFetcher, FetchResult, MAX_RESPONSE_BYTES
from app.ai_engine.schemas.research import SearchRequest, ResearchResponse
from app.ai_engine.search.models import RawSearchResult
from app.ai_engine.search.provider import MockSearchProvider
from app.ai_engine.coordinator import WebResearchEngine


class TestAIEnginePhase1(unittest.TestCase):
    """Unit tests for Phase 1 Web Research Engine components."""

    def setUp(self):
        self.app = create_app("testing")
        self.app.config["AI_ENGINE_ENABLED"] = True
        self.client = self.app.test_client()

    # ─────────────────────────────────────────────────────────────
    # 1. Search Response Normalization
    # ─────────────────────────────────────────────────────────────
    def test_search_response_normalization(self):
        """Verify normalization from raw search results to structured response."""
        mock_raw = [
            RawSearchResult(
                title="AI Trends 2026",
                url="https://example.com/ai-trends",
                snippet="Overview of AI engineering skills.",
                source="example.com",
                published_at="2026-05-01T00:00:00Z",
            )
        ]
        mock_provider = MockSearchProvider(predefined_results=mock_raw)
        
        # Mock web fetcher to return sample HTML
        mock_fetcher = MagicMock()
        mock_fetch_result = MagicMock()
        mock_fetch_result.success = True
        mock_fetch_result.final_url = "https://example.com/ai-trends"
        mock_fetch_result.html = "<html><head><title>AI Trends 2026</title></head><body><p>Deep dive into AI skills.</p></body></html>"
        mock_fetcher.fetch_many.return_value = {"https://example.com/ai-trends": mock_fetch_result}

        engine = WebResearchEngine(search_service=MagicMock(execute_search=mock_provider.search), fetcher=mock_fetcher)
        req = SearchRequest(query="AI engineering skills", max_results=1)
        resp = engine.research(req)

        self.assertIsInstance(resp, ResearchResponse)
        self.assertEqual(resp.query, "AI engineering skills")
        self.assertEqual(resp.total_results, 1)
        result_item = resp.results[0]
        self.assertEqual(result_item.title, "AI Trends 2026")
        self.assertEqual(result_item.url, "https://example.com/ai-trends")
        self.assertIn("Deep dive into AI skills", result_item.content)
        self.assertIsNotNone(result_item.metadata)
        self.assertEqual(result_item.metadata.domain, "example.com")

    # ─────────────────────────────────────────────────────────────
    # 2. Invalid Query Handling
    # ─────────────────────────────────────────────────────────────
    def test_invalid_query_rejection(self):
        """Test missing query, non-string query, and overly long query return 400."""
        # Missing query
        r1 = self.client.post("/api/ai/research/search", json={})
        self.assertEqual(r1.status_code, 400)
        self.assertEqual(r1.get_json()["error"], "validation_error")

        # Query too short (< 2 chars)
        r2 = self.client.post("/api/ai/research/search", json={"query": "a"})
        self.assertEqual(r2.status_code, 400)

        # Overly long query (> 500 chars)
        r3 = self.client.post("/api/ai/research/search", json={"query": "x" * 501})
        self.assertEqual(r3.status_code, 400)

        # Non-string query
        r4 = self.client.post("/api/ai/research/search", json={"query": 12345})
        self.assertEqual(r4.status_code, 400)

    # ─────────────────────────────────────────────────────────────
    # 3. Empty Query Handling
    # ─────────────────────────────────────────────────────────────
    def test_empty_query_rejection(self):
        """Test whitespace-only and empty strings return 400."""
        r1 = self.client.post("/api/ai/research/search", json={"query": ""})
        self.assertEqual(r1.status_code, 400)

        r2 = self.client.post("/api/ai/research/search", json={"query": "   \n\t  "})
        self.assertEqual(r2.status_code, 400)

    # ─────────────────────────────────────────────────────────────
    # 4. URL Validation & Direct SSRF Protection
    # ─────────────────────────────────────────────────────────────
    def test_direct_ssrf_blocking(self):
        """Verify strict SSRF blocker catches private IPs, loopback, metadata, and non-http schemes."""
        # Loopback
        valid, reason, _ = validate_url("http://127.0.0.1:5000/api", resolve_dns=False)
        self.assertFalse(valid)
        self.assertIn("SSRF", reason)

        valid, reason, _ = validate_url("http://localhost:8080", resolve_dns=False)
        self.assertFalse(valid)
        self.assertIn("Blocked hostname", reason)

        # Private RFC1918 subnets
        valid, _, _ = validate_url("http://10.0.0.1/admin", resolve_dns=False)
        self.assertFalse(valid)
        valid, _, _ = validate_url("http://192.168.1.1/router", resolve_dns=False)
        self.assertFalse(valid)
        valid, _, _ = validate_url("http://172.16.0.10", resolve_dns=False)
        self.assertFalse(valid)

        # Cloud metadata service (169.254.169.254)
        valid, reason, _ = validate_url("http://169.254.169.254/latest/meta-data/", resolve_dns=False)
        self.assertFalse(valid)
        self.assertIn("metadata", reason.lower())

        # Cloud metadata hostname
        valid, _, _ = validate_url("http://metadata.google.internal/computeMetadata/v1/", resolve_dns=False)
        self.assertFalse(valid)

        # Internal domain suffix
        valid, _, _ = validate_url("http://internal-service.local", resolve_dns=False)
        self.assertFalse(valid)

        # Unsupported schemes
        valid, reason, _ = validate_url("file:///etc/passwd", resolve_dns=False)
        self.assertFalse(valid)
        self.assertIn("Unsupported scheme", reason)

        valid, _, _ = validate_url("gopher://127.0.0.1:70", resolve_dns=False)
        self.assertFalse(valid)

        # Valid public URL
        valid, _, norm = validate_url("https://example.com/research/paper", resolve_dns=False)
        self.assertTrue(valid)
        self.assertEqual(norm, "https://example.com/research/paper")

    # ─────────────────────────────────────────────────────────────
    # 5. Multi-Hop Redirect SSRF Protection
    # ─────────────────────────────────────────────────────────────
    @patch("requests.Session.get")
    def test_redirect_ssrf_to_localhost_blocked(self, mock_get):
        """Simulate public URL redirecting to localhost (must be blocked)."""
        # First hop: 302 redirect to localhost
        mock_resp_1 = MagicMock()
        mock_resp_1.status_code = 302
        mock_resp_1.headers = {"Location": "http://127.0.0.1:5000/internal"}

        mock_get.return_value = mock_resp_1

        fetcher = WebFetcher()
        result = fetcher.fetch("https://public-site.com/redirect")

        self.assertFalse(result.success)
        self.assertIn("SSRF validation failed", result.error)

    @patch("requests.Session.get")
    def test_redirect_ssrf_to_cloud_metadata_blocked(self, mock_get):
        """Simulate public URL redirecting to AWS/GCP metadata address."""
        mock_resp_1 = MagicMock()
        mock_resp_1.status_code = 301
        mock_resp_1.headers = {"Location": "http://169.254.169.254/latest/user-data"}

        mock_get.return_value = mock_resp_1

        fetcher = WebFetcher()
        result = fetcher.fetch("https://public-site.com/meta-redirect")

        self.assertFalse(result.success)
        self.assertIn("SSRF validation failed", result.error)

    @patch("requests.Session.get")
    def test_redirect_ssrf_to_private_ip_blocked(self, mock_get):
        """Simulate public URL redirecting to 10.0.0.1 private subnet."""
        mock_resp_1 = MagicMock()
        mock_resp_1.status_code = 307
        mock_resp_1.headers = {"Location": "http://10.0.0.1/dashboard"}

        mock_get.return_value = mock_resp_1

        fetcher = WebFetcher()
        result = fetcher.fetch("https://public-site.com/private-redirect")

        self.assertFalse(result.success)
        self.assertIn("SSRF validation failed", result.error)

    # ─────────────────────────────────────────────────────────────
    # 6. Streaming 2MB Size Limit
    # ─────────────────────────────────────────────────────────────
    @patch("app.ai_engine.fetch.web_fetcher.validate_url")
    @patch("requests.Session.get")
    def test_streaming_2mb_size_limit(self, mock_get, mock_validate):
        """Verify fetcher aborts reading once 2MB is reached without crashing."""
        mock_validate.return_value = (True, "Valid", "https://example.com/large")

        # Create mock stream generating 3MB of content in 512KB chunks
        chunk_size = 512 * 1024
        chunks = [b"A" * chunk_size] * 6  # 3 MB total

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.encoding = "utf-8"
        mock_resp.iter_content.return_value = iter(chunks)
        mock_get.return_value = mock_resp

        fetcher = WebFetcher()
        result = fetcher.fetch("https://example.com/large")

        self.assertTrue(result.success)
        self.assertEqual(result.bytes_read, MAX_RESPONSE_BYTES)
        self.assertEqual(len(result.html), MAX_RESPONSE_BYTES)

    # ─────────────────────────────────────────────────────────────
    # 7. Fetch Timeout Handling
    # ─────────────────────────────────────────────────────────────
    @patch("app.ai_engine.fetch.web_fetcher.validate_url")
    @patch("requests.Session.get")
    def test_fetch_timeout_handling(self, mock_get, mock_validate):
        """Verify request timeout is handled gracefully."""
        mock_validate.return_value = (True, "Valid", "https://example.com/slow")
        mock_get.side_effect = requests.Timeout("Connection timed out")

        fetcher = WebFetcher()
        result = fetcher.fetch("https://example.com/slow")

        self.assertFalse(result.success)
        self.assertIn("timed out", result.error.lower())

    # ─────────────────────────────────────────────────────────────
    # 8. Failed Page Handling (404, 500)
    # ─────────────────────────────────────────────────────────────
    @patch("app.ai_engine.fetch.web_fetcher.validate_url")
    @patch("requests.Session.get")
    def test_failed_page_handling(self, mock_get, mock_validate):
        """Verify 404 or 500 error does not crash the pipeline."""
        mock_validate.return_value = (True, "Valid", "https://example.com/missing")
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        fetcher = WebFetcher()
        result = fetcher.fetch("https://example.com/missing")

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 404)
        self.assertIn("HTTP 404", result.error)

    # ─────────────────────────────────────────────────────────────
    # 9. Clean Content & Source Metadata Extraction
    # ─────────────────────────────────────────────────────────────
    def test_content_and_metadata_extraction(self):
        """Verify HTML boilerplate is stripped and metadata/full content are preserved."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>State of Machine Learning 2026</title>
            <meta name="description" content="A comprehensive report on ML frameworks." />
            <meta name="author" content="Dr. Jane Doe" />
            <meta property="article:published_time" content="2026-03-15T10:00:00Z" />
            <link rel="canonical" href="https://example.com/ml-2026" />
            <style>body { font-family: sans-serif; }</style>
            <script>console.log("analytics");</script>
        </head>
        <body>
            <header><nav><a href="/">Home</a></nav></header>
            <article>
                <h1>Machine Learning in 2026</h1>
                <p>Transformer models have expanded into multimodal architectures.</p>
                <p>Retrieval-Augmented Generation remains the primary factual grounding pattern.</p>
            </article>
            <footer><p>&copy; 2026 Example Corp</p></footer>
        </body>
        </html>
        """
        extracted = ContentExtractor.extract(html, "https://example.com/ml-2026")

        self.assertEqual(extracted.title, "State of Machine Learning 2026")
        self.assertEqual(extracted.metadata.author, "Dr. Jane Doe")
        self.assertEqual(extracted.metadata.description, "A comprehensive report on ML frameworks.")
        self.assertEqual(extracted.metadata.published_at, "2026-03-15T10:00:00Z")
        self.assertEqual(extracted.metadata.canonical_url, "https://example.com/ml-2026")
        self.assertEqual(extracted.metadata.domain, "example.com")

        # Assert scripts, styles, nav, footer were stripped
        self.assertNotIn("console.log", extracted.text)
        self.assertNotIn("font-family", extracted.text)
        self.assertNotIn("Home", extracted.text)
        self.assertNotIn("Example Corp", extracted.text)

        # Assert core content is fully preserved
        self.assertIn("Machine Learning in 2026", extracted.text)
        self.assertIn("Transformer models have expanded", extracted.text)
        self.assertIn("Retrieval-Augmented Generation remains", extracted.text)

    # ─────────────────────────────────────────────────────────────
    # 10. URL Normalization & Deduplication
    # ─────────────────────────────────────────────────────────────
    def test_url_normalization_and_deduplication(self):
        """Test removal of tracking params, fragments, and deduplication."""
        raw_urls = [
            "https://EXAMPLE.com/path/?utm_source=twitter&utm_medium=social#section1",
            "https://example.com/path",
            "https://example.com/path/",
            "https://other.org/article?fbclid=12345",
        ]
        deduped = deduplicate_urls(raw_urls)

        # First 3 normalize to https://example.com/path
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0], "https://example.com/path")
        self.assertEqual(deduped[1], "https://other.org/article")

    # ─────────────────────────────────────────────────────────────
    # 11. Search Provider Failure Handling
    # ─────────────────────────────────────────────────────────────
    def test_search_provider_failure_handling(self):
        """Verify unhandled provider errors produce clean 500 responses."""
        failing_provider = MockSearchProvider(should_fail=True)
        engine = WebResearchEngine(search_service=MagicMock(execute_search=failing_provider.search))

        with patch("app.routes.ai_engine._engine", engine):
            resp = self.client.post("/api/ai/research/search", json={"query": "test query"})
            self.assertEqual(resp.status_code, 500)
            data = resp.get_json()
            self.assertEqual(data["error"], "research_failed")

    # ─────────────────────────────────────────────────────────────
    # 12. Feature Flag (AI_ENGINE_ENABLED)
    # ─────────────────────────────────────────────────────────────
    def test_feature_flag_disabled(self):
        """Verify endpoint returns 503 when AI_ENGINE_ENABLED is False."""
        self.app.config["AI_ENGINE_ENABLED"] = False
        resp = self.client.post("/api/ai/research/search", json={"query": "AI skills"})
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.get_json()["error"], "ai_engine_disabled")


class TestDuckDuckGoParserAttributeOrder(unittest.TestCase):
    """
    Regression tests for DuckDuckGoSearchProvider._parse_html().

    These tests are 100% offline (no HTTP). They verify the parser extracts
    results correctly regardless of HTML attribute ordering in anchor tags.

    Root cause fixed: DuckDuckGo renders:
        <a rel="nofollow" class="result__a" href="...">
    The old regex assumed class= was the FIRST attribute, causing zero matches.
    """

    def _make_provider(self):
        from app.ai_engine.search.provider import DuckDuckGoSearchProvider
        return DuckDuckGoSearchProvider()

    def _make_result_block(self, anchor_tag: str, snippet_html: str = "") -> str:
        """Wrap an anchor tag in the minimal DuckDuckGo result div structure."""
        snippet_part = snippet_html or '<a class="result__snippet" href="https://example.com">A snippet.</a>'
        return (
            '<div class="result results_links results_links_deep web-result ">'
            '<div class="links_main links_deep result__body">'
            '<h2 class="result__title">' + anchor_tag + '</h2>'
            + snippet_part +
            "</div></div>"
        )

    # ─────────────────────────────────────────────────────────────
    # Case A: class is first attribute (original / pre-bug-fix order)
    # ─────────────────────────────────────────────────────────────
    def test_parser_case_a_class_first(self):
        """Parser extracts result when class='result__a' is the first attribute."""
        anchor = '<a class="result__a" href="https://example.com/case-a">Case A Title</a>'
        html = self._make_result_block(anchor)
        provider = self._make_provider()
        results = provider._parse_html(html, max_results=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Case A Title")
        self.assertEqual(results[0].url, "https://example.com/case-a")

    # ─────────────────────────────────────────────────────────────
    # Case B: rel="nofollow" BEFORE class (live DuckDuckGo format)
    # ─────────────────────────────────────────────────────────────
    def test_parser_case_b_rel_before_class(self):
        """Parser extracts result when rel='nofollow' precedes class='result__a'."""
        anchor = '<a rel="nofollow" class="result__a" href="https://example.com/case-b">Case B Title</a>'
        html = self._make_result_block(anchor)
        provider = self._make_provider()
        results = provider._parse_html(html, max_results=5)

        self.assertEqual(len(results), 1, "Expected 1 result but got 0 — attribute-order bug not fixed")
        self.assertEqual(results[0].title, "Case B Title")
        self.assertEqual(results[0].url, "https://example.com/case-b")

    # ─────────────────────────────────────────────────────────────
    # Case C: href BEFORE class (defensive future-proofing)
    # ─────────────────────────────────────────────────────────────
    def test_parser_case_c_href_before_class(self):
        """Parser extracts result when href precedes class='result__a'."""
        anchor = '<a href="https://example.com/case-c" class="result__a">Case C Title</a>'
        html = self._make_result_block(anchor)
        provider = self._make_provider()
        results = provider._parse_html(html, max_results=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Case C Title")
        self.assertEqual(results[0].url, "https://example.com/case-c")

    # ─────────────────────────────────────────────────────────────
    # Multiple results mixed attribute orders
    # ─────────────────────────────────────────────────────────────
    def test_parser_multiple_results_mixed_attribute_order(self):
        """Parser handles a full results page with mixed attribute orders correctly."""
        html = (
            # Result 1: class first
            '<div class="result results_links results_links_deep web-result ">'
            '<div class="links_main links_deep result__body">'
            '<h2 class="result__title">'
            '<a class="result__a" href="https://example.com/r1">Result One</a>'
            "</h2>"
            '<a class="result__snippet" href="https://example.com/r1">Snippet one.</a>'
            "</div></div>"
            # Result 2: rel before class (live DDG format)
            '<div class="result results_links results_links_deep web-result ">'
            '<div class="links_main links_deep result__body">'
            '<h2 class="result__title">'
            '<a rel="nofollow" class="result__a" href="https://example.com/r2">Result Two</a>'
            "</h2>"
            '<a class="result__snippet" href="https://example.com/r2">Snippet two.</a>'
            "</div></div>"
            # Result 3: data attribute before class
            '<div class="result results_links results_links_deep web-result ">'
            '<div class="links_main links_deep result__body">'
            '<h2 class="result__title">'
            '<a data-rank="3" class="result__a" href="https://example.com/r3">Result Three</a>'
            "</h2>"
            '<a class="result__snippet" href="https://example.com/r3">Snippet three.</a>'
            "</div></div>"
        )
        provider = self._make_provider()
        results = provider._parse_html(html, max_results=10)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].title, "Result One")
        self.assertEqual(results[1].title, "Result Two")
        self.assertEqual(results[2].title, "Result Three")
        for r in results:
            self.assertTrue(r.url.startswith("https://example.com/"))
            self.assertNotEqual(r.snippet, "")

    # ─────────────────────────────────────────────────────────────
    # Snippet attribute-order variation
    # ─────────────────────────────────────────────────────────────
    def test_parser_snippet_attribute_order_variation(self):
        """Snippet is extracted correctly when snippet anchor has extra attributes."""
        anchor = '<a rel="nofollow" class="result__a" href="https://example.com/snip-test">Snip Title</a>'
        snippet = '<a href="https://example.com/snip-test" class="result__snippet">Custom snippet text.</a>'
        html = self._make_result_block(anchor, snippet_html=snippet)
        provider = self._make_provider()
        results = provider._parse_html(html, max_results=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Snip Title")
        self.assertIn("Custom snippet text", results[0].snippet)

    # ─────────────────────────────────────────────────────────────
    # max_results is respected
    # ─────────────────────────────────────────────────────────────
    def test_parser_max_results_respected(self):
        """Parser honours max_results even when more results are available."""
        block = (
            '<div class="result results_links results_links_deep web-result ">'
            '<div class="links_main links_deep result__body">'
            '<h2 class="result__title">'
            '<a rel="nofollow" class="result__a" href="https://example.com/x">Title X</a>'
            "</h2>"
            '<a class="result__snippet" href="https://example.com/x">Snip X.</a>'
            "</div></div>"
        )
        html = block * 10  # 10 identical blocks
        provider = self._make_provider()
        results = provider._parse_html(html, max_results=3)
        self.assertEqual(len(results), 3)


class TestCoordinatorURLLookupRegression(unittest.TestCase):
    """
    Regression tests for coordinator URL lookup mismatch and canonical deduplication.
    
    Verifies that:
    1. Identical raw and normalized URLs fetch and attach properly.
    2. Trailing slash differences between raw and normalized URL do not lose fetch results.
    3. Tracking parameters stripped during normalization do not cause fetch lookup misses.
    4. Multiple search results resolving to the same normalized URL are deduplicated.
    5. Successful fetch results whose normalized URL differs from raw URL are correctly attached.
    6. Content is attached to the correct search result when multiple distinct pages are returned.
    7. Raw URLs are preserved in the final API response.
    """

    def _make_engine(self, raw_results, fetch_results_dict):
        mock_provider = MockSearchProvider(predefined_results=raw_results)
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_many.side_effect = lambda urls: {
            u: fetch_results_dict[u] for u in urls if u in fetch_results_dict
        }
        engine = WebResearchEngine(
            search_service=MagicMock(execute_search=mock_provider.search),
            fetcher=mock_fetcher,
        )
        return engine, mock_fetcher

    def test_identical_raw_and_normalized_url(self):
        """Regression 1: Identical raw and normalized URL attaches fetched content."""
        url = "https://example.com/clean-article"
        raw = [RawSearchResult(title="Clean Article", url=url, snippet="Clean snippet")]
        fetch_res = FetchResult(
            original_url=url,
            final_url=url,
            status_code=200,
            html="<html><head><title>Clean Article</title></head><body><p>Clean body content</p></body></html>",
            success=True,
        )
        engine, mock_fetcher = self._make_engine(raw, {url: fetch_res})
        resp = engine.research(SearchRequest(query="clean", fetch_content=True))

        mock_fetcher.fetch_many.assert_called_once_with([url])
        self.assertEqual(resp.total_results, 1)
        self.assertEqual(resp.results[0].url, url)
        self.assertEqual(resp.results[0].fetch_status, "fetched")
        self.assertIn("Clean body content", resp.results[0].content)

    def test_trailing_slash_normalization(self):
        """Regression 2: Trailing slash in raw URL is normalized, and fetch result is found."""
        raw_url = "https://example.com/article/"
        norm_url = "https://example.com/article"
        self.assertEqual(normalize_url(raw_url), norm_url)

        raw = [RawSearchResult(title="Slash Article", url=raw_url, snippet="Slash snippet")]
        fetch_res = FetchResult(
            original_url=norm_url,
            final_url=norm_url,
            status_code=200,
            html="<html><head><title>Slash Article</title></head><body><p>Trailing slash content</p></body></html>",
            success=True,
        )
        engine, mock_fetcher = self._make_engine(raw, {norm_url: fetch_res})
        resp = engine.research(SearchRequest(query="slash", fetch_content=True))

        mock_fetcher.fetch_many.assert_called_once_with([norm_url])
        self.assertEqual(resp.total_results, 1)
        # Original raw URL is preserved in the response
        self.assertEqual(resp.results[0].url, raw_url)
        self.assertEqual(resp.results[0].fetch_status, "fetched")
        self.assertIn("Trailing slash content", resp.results[0].content)

    def test_tracking_parameter_removal(self):
        """Regression 3: Tracking params stripped during normalization do not cause lookup miss."""
        raw_url = "https://example.com/article/?utm_source=ddg&utm_campaign=winter&id=5"
        norm_url = "https://example.com/article?id=5"
        self.assertEqual(normalize_url(raw_url), norm_url)

        raw = [RawSearchResult(title="Tracking Article", url=raw_url, snippet="Tracking snippet")]
        fetch_res = FetchResult(
            original_url=norm_url,
            final_url=norm_url,
            status_code=200,
            html="<html><head><title>Tracking Article</title></head><body><p>Cleaned tracking content</p></body></html>",
            success=True,
        )
        engine, mock_fetcher = self._make_engine(raw, {norm_url: fetch_res})
        resp = engine.research(SearchRequest(query="tracking", fetch_content=True))

        mock_fetcher.fetch_many.assert_called_once_with([norm_url])
        self.assertEqual(resp.total_results, 1)
        # Original URL with tracking params preserved in API response
        self.assertEqual(resp.results[0].url, raw_url)
        self.assertEqual(resp.results[0].fetch_status, "fetched")
        self.assertIn("Cleaned tracking content", resp.results[0].content)

    def test_multiple_results_with_normalized_duplicates(self):
        """Regression 4: Multiple raw results that normalize to the same canonical URL are deduplicated."""
        raw1 = RawSearchResult(title="Result 1", url="https://example.com/page/?utm_source=ddg", snippet="Snippet 1")
        raw2 = RawSearchResult(title="Result 2", url="https://example.com/page/?utm_medium=email", snippet="Snippet 2")
        raw3 = RawSearchResult(title="Result 3", url="https://example.com/page/", snippet="Snippet 3")
        raw4 = RawSearchResult(title="Result 4", url="https://example.com/other-page", snippet="Snippet 4")

        norm_page = "https://example.com/page"
        norm_other = "https://example.com/other-page"

        fetch_page = FetchResult(
            original_url=norm_page,
            final_url=norm_page,
            status_code=200,
            html="<html><head><title>Page</title></head><body><p>Page Content</p></body></html>",
            success=True,
        )
        fetch_other = FetchResult(
            original_url=norm_other,
            final_url=norm_other,
            status_code=200,
            html="<html><head><title>Other</title></head><body><p>Other Content</p></body></html>",
            success=True,
        )

        engine, mock_fetcher = self._make_engine(
            [raw1, raw2, raw3, raw4],
            {norm_page: fetch_page, norm_other: fetch_other},
        )
        resp = engine.research(SearchRequest(query="dedup", fetch_content=True, max_results=5))

        # Only 2 unique canonical URLs should be fetched
        mock_fetcher.fetch_many.assert_called_once_with([norm_page, norm_other])
        self.assertEqual(resp.total_results, 2)
        self.assertEqual(resp.results[0].title, "Result 1")
        self.assertEqual(resp.results[0].url, "https://example.com/page/?utm_source=ddg")
        self.assertIn("Page Content", resp.results[0].content)
        self.assertEqual(resp.results[1].title, "Result 4")
        self.assertEqual(resp.results[1].url, "https://example.com/other-page")
        self.assertIn("Other Content", resp.results[1].content)

    def test_successful_fetch_normalized_differs_from_raw(self):
        """Regression 5: Successful fetch whose normalized URL differs from raw URL attaches correctly."""
        raw_url = "https://example.com/blog/ai-future/?utm_source=ddg&ref=partner#heading"
        norm_url = "https://example.com/blog/ai-future"
        self.assertNotEqual(raw_url, norm_url)
        self.assertEqual(normalize_url(raw_url), norm_url)

        raw = [RawSearchResult(title="AI Future", url=raw_url, snippet="AI Future snippet")]
        fetch_res = FetchResult(
            original_url=norm_url,
            final_url=norm_url,
            status_code=200,
            html="<html><head><title>AI Future</title></head><body><p>Full AI Future article text.</p></body></html>",
            success=True,
        )
        engine, mock_fetcher = self._make_engine(raw, {norm_url: fetch_res})
        resp = engine.research(SearchRequest(query="ai future", fetch_content=True))

        mock_fetcher.fetch_many.assert_called_once_with([norm_url])
        self.assertEqual(resp.total_results, 1)
        self.assertEqual(resp.results[0].url, raw_url)
        self.assertEqual(resp.results[0].fetch_status, "fetched")
        self.assertIn("Full AI Future article text", resp.results[0].content)

    def test_content_attached_to_correct_search_result(self):
        """Requirement 9: Content is attached to the correct search result when multiple distinct pages are returned."""
        raw_a = RawSearchResult(title="Site A", url="https://site-a.com/guide/?utm_source=a", snippet="Snippet A")
        raw_b = RawSearchResult(title="Site B", url="https://site-b.com/tutorial/", snippet="Snippet B")
        norm_a = "https://site-a.com/guide"
        norm_b = "https://site-b.com/tutorial"

        fetch_a = FetchResult(
            original_url=norm_a,
            final_url=norm_a,
            status_code=200,
            html="<html><body><p>Exclusive content for Site A.</p></body></html>",
            success=True,
        )
        fetch_b = FetchResult(
            original_url=norm_b,
            final_url=norm_b,
            status_code=200,
            html="<html><body><p>Exclusive content for Site B.</p></body></html>",
            success=True,
        )

        engine, _ = self._make_engine([raw_a, raw_b], {norm_a: fetch_a, norm_b: fetch_b})
        resp = engine.research(SearchRequest(query="guides", fetch_content=True))

        self.assertEqual(resp.total_results, 2)
        # Site A must have Site A's content and NOT Site B's content
        self.assertIn("Exclusive content for Site A", resp.results[0].content)
        self.assertNotIn("Exclusive content for Site B", resp.results[0].content)
        self.assertEqual(resp.results[0].url, raw_a.url)

        # Site B must have Site B's content and NOT Site A's content
        self.assertIn("Exclusive content for Site B", resp.results[1].content)
        self.assertNotIn("Exclusive content for Site A", resp.results[1].content)
        self.assertEqual(resp.results[1].url, raw_b.url)


if __name__ == "__main__":
    unittest.main()
