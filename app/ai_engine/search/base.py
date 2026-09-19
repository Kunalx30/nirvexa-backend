"""
app/ai_engine/search/base.py
Abstract base class for search providers.
"""
from abc import ABC, abstractmethod
from typing import List
from app.ai_engine.search.models import RawSearchResult


class SearchProvider(ABC):
    """Abstract interface for all web search providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier name."""
        pass

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[RawSearchResult]:
        """
        Execute a search query and return a list of raw search results.
        Raises an exception if the provider experiences a fatal error.
        """
        pass
