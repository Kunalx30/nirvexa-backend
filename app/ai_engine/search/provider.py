"""
app/ai_engine/search/provider.py
Search provider implementations:
- DuckDuckGoSearchProvider (Live search, zero API key required)
- MockSearchProvider (Deterministic in-memory search for unit testing)
"""
import re
import logging
import urllib.parse
from typing import List, Optional
import requests
from app.ai_engine.search.base import SearchProvider
from app.ai_engine.search.models import RawSearchResult

logger = logging.getLogger(__name__)


class MockSearchProvider(SearchProvider):
    """
    In-memory deterministic mock search provider.
    Used for unit testing without live network connections.
    """

    def __init__(self, predefined_results: Optional[List[RawSearchResult]] = None, should_fail: bool = False):
        self._predefined = predefined_results
        self._should_fail = should_fail

    @property
    def name(self) -> str:
        return "mock"

    def set_predefined_results(self, results: List[RawSearchResult]):
        self._predefined = results

    def set_should_fail(self, should_fail: bool):
        self._should_fail = should_fail

    def search(self, query: str, max_results: int = 5) -> List[RawSearchResult]:
        if self._should_fail:
            raise RuntimeError("Simulated search provider failure")

        if self._predefined is not None:
            return self._predefined[:max_results]

        # Generate deterministic synthetic results
        results = []
        for i in range(1, max_results + 1):
            results.append(
                RawSearchResult(
                    title=f"Sample Result {i} for '{query}'",
                    url=f"https://example.com/research/{i}?q={urllib.parse.quote(query)}",
                    snippet=f"Detailed snippet discussing '{query}' with relevant industry facts and figures ({i}).",
                    source="example.com",
                    published_at=None,
                )
            )
        return results


class DuckDuckGoSearchProvider(SearchProvider):
    """
    Live web search provider using DuckDuckGo HTML API.
    Does not require an external API key.
    """

    ENDPOINT = "https://html.duckduckgo.com/html/"
    DEFAULT_TIMEOUT = 10  # seconds

    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        self.timeout = timeout
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
        }

    @property
    def name(self) -> str:
        return "duckduckgo"

    def search(self, query: str, max_results: int = 5) -> List[RawSearchResult]:
        try:
            resp = requests.post(
                self.ENDPOINT,
                data={"q": query, "b": ""},
                headers=self.headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            # Verify we got HTML — DDG could return a rate-limit or CAPTCHA page
            content_type = resp.headers.get("Content-Type", "")
            if "html" not in content_type.lower():
                logger.warning(
                    "[DuckDuckGoSearchProvider] Unexpected Content-Type %r for query %r",
                    content_type, query,
                )
                return []
            return self._parse_html(resp.text, max_results)
        except requests.RequestException as e:
            logger.error(
                "[DuckDuckGoSearchProvider] Request failed for query %r: %s",
                query, e,
            )
            raise RuntimeError(f"Search provider error: {e}") from e

    def _parse_html(self, html: str, max_results: int) -> List[RawSearchResult]:
        results: List[RawSearchResult] = []

        def _is_valid_url(url: str) -> bool:
            """Quick scheme check — reject javascript:, mailto:, empty, etc."""
            return bool(url) and url.startswith(("http://", "https://"))

        def _append_if_valid(title: str, url: str, snippet: str) -> bool:
            """Validate and append a raw result. Returns True if added."""
            if not _is_valid_url(url):
                return False
            if not title.strip():
                return False
            domain = urllib.parse.urlparse(url).netloc
            results.append(
                RawSearchResult(
                    title=title.strip(),
                    url=url,
                    snippet=snippet.strip(),
                    source=domain or "web",
                )
            )
            return True

        # Find result containers: <div class="result ..."> or links with class result__a / result__snippet
        # DuckDuckGo HTML layout (actual live response):
        # <a rel="nofollow" class="result__a" href="https://...">TITLE</a>
        # <a class="result__snippet" href="...">SNIPPET</a>
        # NOTE: Attribute order can vary — regexes must NOT assume class= is first.
        
        # Regex matching result blocks
        result_blocks = re.findall(
            r'<div class="result[^"]*">([\s\S]*?)</div>\s*</div>',
            html,
            re.IGNORECASE
        )

        if not result_blocks:
            # Fallback block parsing.
            # Use \b before class= so attribute order (e.g. rel="nofollow" appearing first)
            # does not prevent matching.
            result_blocks = re.findall(
                r'<h2 class="result__title">([\s\S]*?)</h2>[\s\S]*?<a\b[^>]*\bclass="result__snippet[^"]*"[^>]*>([\s\S]*?)</a>',
                html,
                re.IGNORECASE
            )
            for title_part, snippet_part in result_blocks:
                link_match = re.search(r'href="([^"]+)"', title_part)
                raw_url = link_match.group(1) if link_match else ""
                clean_url = self._extract_target_url(raw_url)
                clean_title = re.sub(r'<[^>]+>', '', title_part).strip()
                clean_snippet = re.sub(r'<[^>]+>', '', snippet_part).strip()
                _append_if_valid(clean_title, clean_url, clean_snippet)
                if len(results) >= max_results:
                    break
            return results

        for block in result_blocks:
            # Two-pass extraction: attribute order is fully arbitrary.
            # Pass 1 — find the <a> element that has class="result__a" anywhere in the tag,
            # using a lookahead so the class check does not consume characters.
            anchor_match = re.search(
                r'<a\b(?=[^>]*\bclass="result__a")[^>]*>([\s\S]*?)</a>',
                block, re.IGNORECASE
            )
            snippet_match = re.search(
                r'<a\b[^>]*\bclass="result__snippet"[^>]*>([\s\S]*?)</a>',
                block, re.IGNORECASE
            )

            if anchor_match:
                raw_title = anchor_match.group(1)
                # Pass 2 — extract href from the matched opening tag only (not the content)
                opening_tag = re.search(
                    r'<a\b(?=[^>]*\bclass="result__a")[^>]*>',
                    block, re.IGNORECASE
                )
                if not opening_tag:
                    continue
                href_in_tag = re.search(r'\bhref="([^"]+)"', opening_tag.group(0), re.IGNORECASE)
                if not href_in_tag:
                    continue

                raw_url = href_in_tag.group(1)
                raw_snippet = snippet_match.group(1) if snippet_match else ""

                clean_url = self._extract_target_url(raw_url)
                clean_title = re.sub(r'<[^>]+>', '', raw_title).strip()
                clean_snippet = re.sub(r'<[^>]+>', '', raw_snippet).strip()

                _append_if_valid(clean_title, clean_url, clean_snippet)

            if len(results) >= max_results:
                break

        return results

    @staticmethod
    def _extract_target_url(raw_url: str) -> str:
        """Extract true destination from DuckDuckGo redirect wrapper if present."""
        if not raw_url:
            return ""
        if "uddg=" in raw_url:
            parsed = urllib.parse.urlparse(raw_url)
            qs = urllib.parse.parse_qs(parsed.query)
            if "uddg" in qs and qs["uddg"]:
                return qs["uddg"][0]
        if raw_url.startswith("//"):
            return "https:" + raw_url
        return raw_url
