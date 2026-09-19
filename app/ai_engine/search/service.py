"""
app/ai_engine/search/service.py
Search service providing provider instantiation and dispatch.
"""
import os
import logging
from typing import Optional
from app.ai_engine.search.base import SearchProvider
from app.ai_engine.search.provider import DuckDuckGoSearchProvider, MockSearchProvider

logger = logging.getLogger(__name__)


class SearchService:
    """Service managing search providers and query execution."""

    def __init__(self, provider: Optional[SearchProvider] = None):
        self._provider = provider or self._resolve_provider()

    @property
    def provider(self) -> SearchProvider:
        return self._provider

    @staticmethod
    def _resolve_provider() -> SearchProvider:
        provider_name = os.getenv("AI_ENGINE_SEARCH_PROVIDER", "duckduckgo").strip().lower()

        if provider_name == "mock":
            logger.info("[SearchService] Using MockSearchProvider")
            return MockSearchProvider()

        # Default to DuckDuckGo (zero credentials required)
        logger.info("[SearchService] Using DuckDuckGoSearchProvider")
        return DuckDuckGoSearchProvider()

    def execute_search(self, query: str, max_results: int = 5):
        """Execute search using active provider."""
        return self._provider.search(query=query, max_results=max_results)
