"""
app/ai_engine/retrieval/reranker/factory.py
Factory for creating and resolving BaseReranker instances based on configuration.
Supports 'keyword' (default) and 'multi_signal'.
"""
import logging
from typing import Optional, Any

from app.ai_engine.retrieval.reranker.base import BaseReranker
from app.ai_engine.retrieval.reranker.keyword import KeywordReranker
from app.ai_engine.retrieval.reranker.multi_signal import MultiSignalReranker

logger = logging.getLogger(__name__)


def _get_config(key: str, default: Any) -> Any:
    try:
        from flask import current_app
        if current_app and current_app.config:
            return current_app.config.get(key, default)
    except (ImportError, RuntimeError):
        pass
    from config import Config
    return getattr(Config, key, default)


class RerankerFactory:
    """
    Factory resolving BaseReranker instances.
    """

    @staticmethod
    def get_reranker(
        reranker_type: Optional[str] = None,
        min_relevance_score: Optional[float] = None,
        filter_low_relevance: bool = True,
        **kwargs,
    ) -> BaseReranker:
        rtype = (
            reranker_type
            or _get_config("AI_ENGINE_RERANKER_TYPE", "keyword")
        ).lower().strip()

        raw_min = (
            min_relevance_score
            if min_relevance_score is not None
            else _get_config("AI_ENGINE_MIN_RELEVANCE_SCORE", 0.10)
        )
        min_score = max(0.0, min(1.0, float(raw_min)))

        if rtype == "multi_signal":
            return MultiSignalReranker(
                min_relevance_score=min_score,
                filter_low_relevance=filter_low_relevance,
                **kwargs,
            )

        return KeywordReranker(
            min_relevance_score=min_score,
            filter_low_relevance=filter_low_relevance,
        )
