"""
app/ai_engine/schemas/research.py
Pydantic schemas and dataclasses for the Web Research Engine.
"""
from datetime import datetime, timezone
from typing import Optional, List, Any, Literal
from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    """Input payload for web research search."""
    query: str = Field(..., description="The search query to research.")
    max_results: int = Field(default=5, ge=1, le=10, description="Max search results to retrieve (1-10).")
    fetch_content: bool = Field(default=True, description="Whether to fetch and extract full webpage text.")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Query must be a string.")
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Query cannot be empty or whitespace only.")
        if len(trimmed) < 2:
            raise ValueError("Query must be at least 2 characters long.")
        if len(trimmed) > 500:
            raise ValueError("Query must not exceed 500 characters.")
        return trimmed


class SourceMetadata(BaseModel):
    """Extracted metadata about a source webpage."""
    title: str = ""
    description: Optional[str] = None
    author: Optional[str] = None
    published_at: Optional[str] = None
    canonical_url: Optional[str] = None
    domain: str = ""
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# Fetch status values for per-result diagnostics.
# "fetched"   — page content successfully retrieved and extracted.
# "skipped"   — fetch_content=False; not attempted.
# "blocked"   — SSRF validation rejected the URL.
# "error"     — HTTP error, timeout, SSL/connection failure.
# "empty"     — fetch succeeded but response was empty or non-HTML.
FetchStatus = Literal["fetched", "skipped", "blocked", "error", "empty"]


class WebSearchResult(BaseModel):
    """Normalized web research result matching user specification."""
    title: str
    url: str
    snippet: str
    source: str
    published_at: Optional[str] = None
    content: str = ""
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Optional[SourceMetadata] = None
    # Per-result fetch diagnostics (backwards-compatible addition)
    fetch_status: FetchStatus = "skipped"
    fetch_error: Optional[str] = None


# Search-level status distinguishes genuine no-results from provider failures.
# "success"        — provider returned results (>0).
# "no_results"     — provider returned zero results without error.
# "provider_error" — provider raised an exception; results may be empty.
SearchStatus = Literal["success", "no_results", "provider_error"]


class ResearchResponse(BaseModel):
    """Outermost response structure."""
    query: str
    results: List[WebSearchResult]
    total_results: int
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Distinguishes empty-because-no-results from empty-because-error
    search_status: SearchStatus = "success"
    # Human-readable status message (empty string on success)
    status_message: str = ""
