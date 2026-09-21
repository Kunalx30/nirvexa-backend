"""
app/ai_engine/retrieval/reranker/base.py
Abstract interface for candidate evidence rerankers.
Enables pluggable ranking strategies (Keyword, BM25, and future Vector/Embedding models).
"""
from abc import ABC, abstractmethod
from typing import List
from app.ai_engine.retrieval.schemas import EvidenceItem


class BaseReranker(ABC):
    """
    Abstract interface for relevance rerankers.
    Implementations must be deterministic where possible and normalize
    scores to the range [0.0, 1.0].
    """

    @abstractmethod
    def rank(self, query: str, items: List[EvidenceItem]) -> List[EvidenceItem]:
        """
        Ranks and scores EvidenceItem candidates against a query.
        Returns the items sorted in descending order of relevance_score.
        """
        pass
