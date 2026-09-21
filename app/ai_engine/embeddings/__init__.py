"""
app/ai_engine/embeddings/__init__.py
Package exports for the AI Engine Phase 5 embedding layer.
"""
from app.ai_engine.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingProviderUnavailableError,
    EmbeddingProviderTimeoutError,
    EmbeddingProviderRateLimitError,
)
from app.ai_engine.embeddings.mock import MockEmbeddingProvider
from app.ai_engine.embeddings.gemini import GeminiEmbeddingProvider
from app.ai_engine.embeddings.openai import OpenAICompatibleEmbeddingProvider
from app.ai_engine.embeddings.ollama import OllamaEmbeddingProvider
from app.ai_engine.embeddings.factory import EmbeddingProviderFactory

__all__ = [
    "BaseEmbeddingProvider",
    "EmbeddingProviderError",
    "EmbeddingProviderUnavailableError",
    "EmbeddingProviderTimeoutError",
    "EmbeddingProviderRateLimitError",
    "MockEmbeddingProvider",
    "GeminiEmbeddingProvider",
    "OpenAICompatibleEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "EmbeddingProviderFactory",
]
