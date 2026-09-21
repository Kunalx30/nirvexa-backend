"""
app/ai_engine/embeddings/factory.py
Factory for resolving and instantiating embedding providers based on configuration.
Supports mock, gemini, openai, and ollama providers.
"""
import os
import logging
from typing import Optional, Dict, Any

from config import Config
from app.ai_engine.embeddings.base import BaseEmbeddingProvider, EmbeddingProviderError
from app.ai_engine.embeddings.mock import MockEmbeddingProvider
from app.ai_engine.embeddings.gemini import GeminiEmbeddingProvider
from app.ai_engine.embeddings.openai import OpenAICompatibleEmbeddingProvider
from app.ai_engine.embeddings.ollama import OllamaEmbeddingProvider

logger = logging.getLogger(__name__)


def _get_config(key: str, default: Any) -> Any:
    try:
        from flask import current_app
        if current_app and current_app.config:
            return current_app.config.get(key, default)
    except (ImportError, RuntimeError):
        pass
    return getattr(Config, key, default)


class EmbeddingProviderFactory:
    """
    Instantiates the appropriate BaseEmbeddingProvider based on configuration.
    """

    @staticmethod
    def create(
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
    ) -> BaseEmbeddingProvider:
        p_name = (provider_name or _get_config("AI_ENGINE_EMBEDDING_PROVIDER", "gemini")).lower().strip()
        dim = dimension or _get_config("AI_ENGINE_EMBEDDING_DIMENSION", 768)
        timeout = timeout_seconds or 15

        if p_name == "mock":
            m_name = model or "mock-embedding"
            return MockEmbeddingProvider(model=m_name, dimension=dim, timeout_seconds=timeout)

        if p_name == "gemini":
            m_name = model or _get_config("AI_ENGINE_EMBEDDING_MODEL", "text-embedding-004")
            api_key = os.getenv("GEMINI_API_KEY")
            return GeminiEmbeddingProvider(api_key=api_key, model=m_name, dimension=dim, timeout_seconds=timeout)

        if p_name == "openai":
            m_name = model or _get_config("AI_ENGINE_EMBEDDING_MODEL", "text-embedding-3-small")
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            return OpenAICompatibleEmbeddingProvider(
                api_key=api_key, base_url=base_url, model=m_name, dimension=dim, timeout_seconds=timeout
            )

        if p_name == "ollama":
            m_name = model or _get_config("AI_ENGINE_EMBEDDING_MODEL", "nomic-embed-text")
            base_url = os.getenv("AI_ENGINE_OLLAMA_BASE_URL", "http://localhost:11434")
            return OllamaEmbeddingProvider(base_url=base_url, model=m_name, dimension=dim, timeout_seconds=timeout)

        raise EmbeddingProviderError(f"Unsupported embedding provider: {p_name!r}")
