"""
app/ai_engine/fetch/web_fetcher.py
Targeted, SSRF-safe HTTP fetcher with streaming size limits and manual redirect validation.
"""
from dataclasses import dataclass
import logging
from typing import Optional, List, Dict
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import requests.exceptions

from app.ai_engine.fetch.url_validator import validate_url, normalize_url

logger = logging.getLogger(__name__)

# Constants
MAX_RESPONSE_BYTES = 2 * 1024 * 1024   # 2 MB hard cap
MAX_REDIRECTS = 5                        # DuckDuckGo URLs may take 2-3 hops
CHUNK_SIZE = 8192                        # 8 KB streaming chunks
DEFAULT_TIMEOUT = 5                      # seconds (both connect and read)
MAX_CONTENT_LENGTH = MAX_RESPONSE_BYTES  # Reject early if Content-Length header exceeds this


@dataclass
class FetchResult:
    """Result of a webpage fetch attempt."""
    original_url: str
    final_url: str
    status_code: int
    html: str
    success: bool
    error: Optional[str] = None
    bytes_read: int = 0


class WebFetcher:
    """
    Targeted HTTP fetcher for web research.
    Enforces SSRF protection on initial request and every redirect hop.
    Streams body up to 2MB limit.
    Uses a per-instance requests.Session for connection pooling.
    Thread-safe for concurrent use with fetch_many().
    """

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
    }

    def __init__(self, timeout: int = DEFAULT_TIMEOUT, max_bytes: int = MAX_RESPONSE_BYTES):
        # Clamp to sane production bounds
        self.timeout = max(1, min(timeout, 30))
        self.max_bytes = max(65536, min(max_bytes, MAX_RESPONSE_BYTES))  # 64 KB – 2 MB
        # NOTE: requests.Session is NOT thread-safe for concurrent .get() calls.
        # fetch_many() uses ThreadPoolExecutor; each thread therefore uses its own
        # Session (created per-call in _fetch_one) to avoid connection-pool races.

    def fetch(self, url: str) -> FetchResult:
        """
        Fetch a single URL safely.
        Public API — creates a fresh session for this fetch.
        """
        session = requests.Session()
        session.headers.update(self.DEFAULT_HEADERS)
        try:
            return self._fetch_with_session(session, url)
        finally:
            session.close()

    def _fetch_with_session(self, session: requests.Session, url: str) -> FetchResult:
        """
        Internal fetch using the provided session.
        - Validates SSRF on initial URL and every redirect target.
        - Manually follows redirects (up to MAX_REDIRECTS).
        - Streams response body up to max_bytes.
        """
        current_url = url
        redirect_count = 0

        while redirect_count <= MAX_REDIRECTS:
            # 1. Validate URL against SSRF (includes DNS resolution)
            is_valid, reason, norm_url = validate_url(current_url)
            if not is_valid:
                logger.warning("[WebFetcher] SSRF blocked url=%s reason=%s", current_url, reason)
                return FetchResult(
                    original_url=url,
                    final_url=current_url,
                    status_code=0,
                    html="",
                    success=False,
                    error=f"SSRF validation failed: {reason}",
                )

            current_url = norm_url or current_url

            response = None
            try:
                # 2. Issue request with stream=True, allow_redirects=False
                response = session.get(
                    current_url,
                    timeout=self.timeout,
                    stream=True,
                    allow_redirects=False,
                )

                # 3. Handle redirects manually — each Location is re-validated
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location", "").strip()
                    response.close()
                    response = None

                    if not location:
                        return FetchResult(
                            original_url=url,
                            final_url=current_url,
                            status_code=0,
                            html="",
                            success=False,
                            error="Redirect with empty Location header",
                        )

                    # Resolve relative redirect URLs against current URL
                    next_url = urllib.parse.urljoin(current_url, location)
                    logger.debug("[WebFetcher] Redirect %d: %s -> %s", redirect_count + 1, current_url, next_url)
                    current_url = next_url
                    redirect_count += 1
                    continue

                # 4. Check HTTP status before reading body
                if response.status_code >= 400:
                    response.close()
                    return FetchResult(
                        original_url=url,
                        final_url=current_url,
                        status_code=response.status_code,
                        html="",
                        success=False,
                        error=f"HTTP {response.status_code}",
                    )

                # 5. Content-Length early abort — don't start streaming oversized responses
                content_length_header = response.headers.get("Content-Length")
                if content_length_header:
                    try:
                        declared_length = int(content_length_header)
                        if declared_length > self.max_bytes:
                            response.close()
                            logger.info(
                                "[WebFetcher] Content-Length %d exceeds limit %d for %s — skipped",
                                declared_length, self.max_bytes, current_url,
                            )
                            return FetchResult(
                                original_url=url,
                                final_url=current_url,
                                status_code=response.status_code,
                                html="",
                                success=False,
                                error=f"Response too large: Content-Length {declared_length} > {self.max_bytes}",
                            )
                    except (ValueError, TypeError):
                        pass  # Malformed Content-Length — continue streaming with hard cap

                # 6. Stream response up to max_bytes (enforced during streaming regardless of headers)
                content_chunks: List[bytes] = []
                total_bytes = 0
                size_capped = False

                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    remaining = self.max_bytes - total_bytes
                    if remaining <= 0:
                        size_capped = True
                        break

                    if len(chunk) > remaining:
                        content_chunks.append(chunk[:remaining])
                        total_bytes += remaining
                        size_capped = True
                        break
                    else:
                        content_chunks.append(chunk)
                        total_bytes += len(chunk)

                if size_capped:
                    logger.info(
                        "[WebFetcher] Reached max size limit (%d bytes) for %s",
                        self.max_bytes, current_url,
                    )

                response.close()
                response = None

                raw_bytes = b"".join(content_chunks)

                # 7. Decode — honour Content-Type charset, fall back to apparent encoding, then utf-8
                encoding = (
                    response.encoding if response is not None else None
                ) or _detect_charset(raw_bytes) or "utf-8"
                try:
                    html_text = raw_bytes.decode(encoding, errors="replace")
                except (LookupError, UnicodeDecodeError):
                    html_text = raw_bytes.decode("utf-8", errors="replace")

                return FetchResult(
                    original_url=url,
                    final_url=current_url,
                    status_code=(response.status_code if response is not None else 200),
                    html=html_text,
                    success=True,
                    bytes_read=total_bytes,
                )

            except requests.exceptions.SSLError as exc:
                logger.warning("[WebFetcher] SSL error for %s: %s", current_url, exc)
                if response is not None:
                    response.close()
                return FetchResult(
                    original_url=url,
                    final_url=current_url,
                    status_code=0,
                    html="",
                    success=False,
                    error=f"SSL error: {exc}",
                )
            except requests.exceptions.ConnectionError as exc:
                logger.warning("[WebFetcher] Connection error for %s: %s", current_url, exc)
                if response is not None:
                    response.close()
                return FetchResult(
                    original_url=url,
                    final_url=current_url,
                    status_code=0,
                    html="",
                    success=False,
                    error=f"Connection error: {exc}",
                )
            except requests.Timeout as exc:
                logger.warning("[WebFetcher] Timeout fetching %s: %s", current_url, exc)
                if response is not None:
                    response.close()
                return FetchResult(
                    original_url=url,
                    final_url=current_url,
                    status_code=0,
                    html="",
                    success=False,
                    error="Request timed out",
                )
            except requests.RequestException as exc:
                logger.warning("[WebFetcher] Request error for %s: %s", current_url, exc)
                if response is not None:
                    response.close()
                return FetchResult(
                    original_url=url,
                    final_url=current_url,
                    status_code=0,
                    html="",
                    success=False,
                    error=str(exc),
                )
            except Exception as exc:
                logger.error("[WebFetcher] Unexpected error for %s: %s", current_url, exc, exc_info=True)
                if response is not None:
                    response.close()
                return FetchResult(
                    original_url=url,
                    final_url=current_url,
                    status_code=0,
                    html="",
                    success=False,
                    error="Unexpected fetch error",
                )

        # Exceeded redirect limit
        return FetchResult(
            original_url=url,
            final_url=current_url,
            status_code=0,
            html="",
            success=False,
            error=f"Exceeded maximum redirect limit ({MAX_REDIRECTS})",
        )

    def fetch_many(self, urls: List[str], max_workers: int = 4) -> Dict[str, FetchResult]:
        """
        Fetch multiple URLs in parallel using ThreadPoolExecutor.
        Each future creates its own Session to avoid thread-safety issues.
        max_workers is clamped to [1, 10].
        """
        results: Dict[str, FetchResult] = {}
        if not urls:
            return results

        # Clamp worker count: never exceed number of URLs or configured max
        effective_workers = min(len(urls), max(1, min(max_workers, 10)))

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_url = {executor.submit(self.fetch, u): u for u in urls}
            for future in as_completed(future_to_url):
                orig_url = future_to_url[future]
                try:
                    results[orig_url] = future.result()
                except Exception as exc:
                    results[orig_url] = FetchResult(
                        original_url=orig_url,
                        final_url=orig_url,
                        status_code=0,
                        html="",
                        success=False,
                        error=str(exc),
                    )
        return results


def _detect_charset(raw_bytes: bytes) -> Optional[str]:
    """
    Lightweight charset sniffer using only the standard library.
    Checks for UTF-8 BOM and common meta charset declarations in the first 1 KB.
    Returns None if no charset can be detected.
    """
    if raw_bytes[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig"
    head = raw_bytes[:1024].lower()
    import re
    m = re.search(rb'charset=["\']?([a-z0-9_-]+)', head)
    if m:
        return m.group(1).decode("ascii", errors="ignore")
    return None
