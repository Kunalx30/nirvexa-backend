"""
app/ai_engine/reasoning/providers/base.py
Abstract Base Class and standard exceptions for all LLM providers.
Prevents leaking provider-specific SDK objects to downstream callers.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from app.ai_engine.reasoning.schemas import LLMGenerationResult


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    def __init__(self, message: str, provider: str = "unknown", retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.retryable = retryable


class ProviderUnavailableError(LLMProviderError):
    """Raised when the LLM provider service is unreachable or offline."""
    pass


class ProviderTimeoutError(LLMProviderError):
    """Raised when an LLM provider request exceeds its timeout."""
    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider=provider, retryable=True)


class ProviderRateLimitError(LLMProviderError):
    """Raised when an LLM provider rate limit / quota is hit."""
    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider=provider, retryable=True)


class BaseLLMProvider(ABC):
    """
    Abstract interface for LLM text generation.
    All providers must implement `generate` returning a clean LLMGenerationResult.
    """

    def __init__(
        self,
        model: str,
        timeout_seconds: int = 30,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    @abstractmethod
    def name(self) -> str:
        """The canonical name identifier for this provider."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMGenerationResult:
        """
        Executes text generation.
        Returns:
            LLMGenerationResult containing clean text and metadata.
        Raises:
            LLMProviderError (or subclass) on failure.
        """
        pass
