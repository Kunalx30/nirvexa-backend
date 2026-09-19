"""Fetch package for AI Engine."""
from app.ai_engine.fetch.url_validator import (
    validate_url,
    normalize_url,
    deduplicate_urls,
    is_ip_blocked,
)
from app.ai_engine.fetch.web_fetcher import (
    WebFetcher,
    FetchResult,
    MAX_RESPONSE_BYTES,
)

__all__ = [
    "validate_url",
    "normalize_url",
    "deduplicate_urls",
    "is_ip_blocked",
    "WebFetcher",
    "FetchResult",
    "MAX_RESPONSE_BYTES",
]
