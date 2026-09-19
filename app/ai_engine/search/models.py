"""
app/ai_engine/search/models.py
Intermediate data models for search providers.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class RawSearchResult:
    """Raw search result returned by a search provider before fetching content."""
    title: str
    url: str
    snippet: str
    source: str = ""
    published_at: Optional[str] = None
