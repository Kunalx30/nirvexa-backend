"""
app/ai_engine/embeddings/base.py
Abstract base class and exceptions for all embedding providers.
Ensures uniform interface across Mock, Gemini, OpenAI, and Ollama providers.
"""
from abc import ABC, abstractmethod
from typing import List, Optional


class EmbeddingProviderError(Exception):
    """Base exception for embedding provider failures."""
    def __init__(self, message: str, provider: str = "unknown", retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.retryable = retryable


class EmbeddingProviderUnavailableError(EmbeddingProviderError):
    """Raised when an embedding provider service is unreachable."""
    pass


class EmbeddingProviderTimeoutError(EmbeddingProviderError):
    """Raised when an embedding provider exceeds its timeout."""
    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider=provider, retryable=True)


class EmbeddingProviderRateLimitError(EmbeddingProviderError):
    """Raised when an embedding provider rate limit / quota is exceeded."""
    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider=provider, retryable=True)


class BaseEmbeddingProvider(ABC):
    """
    Abstract interface for vector embedding generation.
    All providers must implement embed_text and embed_batch.
    """

    def __init__(self, model: str, dimension: int = 768, timeout_seconds: int = 15):
        self.model = model
        self.dimension = dimension
        self.timeout_seconds = timeout_seconds

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical name identifier of the provider."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generates a vector embedding for a single text string.
        Returns:
            List of floats representing the embedding vector.
        Raises:
            EmbeddingProviderError (or subclass) on failure.
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generates vector embeddings for a list of text strings.
        Returns:
            List of embedding vectors corresponding to the input texts.
        Raises:
            EmbeddingProviderError (or subclass) on failure.
        """
        pass
