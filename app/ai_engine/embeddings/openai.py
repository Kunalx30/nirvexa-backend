"""
app/ai_engine/embeddings/openai.py
OpenAI-compatible embedding provider utilizing pre-installed openai SDK.
Default model: text-embedding-3-small (1536 dimensions).
Also compatible with local OpenAI-compatible embedding servers (vLLM, Ollama, etc.).
"""
import logging
from typing import List, Optional

from app.ai_engine.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingProviderTimeoutError,
    EmbeddingProviderRateLimitError,
    EmbeddingProviderUnavailableError,
)

logger = logging.getLogger(__name__)


class OpenAICompatibleEmbeddingProvider(BaseEmbeddingProvider):
    """
    Produces vector embeddings using any OpenAI-compatible API endpoint.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        timeout_seconds: int = 15,
    ):
        super().__init__(model=model, dimension=dimension, timeout_seconds=timeout_seconds)
        self.api_key = api_key
        self.base_url = base_url

    @property
    def name(self) -> str:
        return "openai"

    def _get_client(self):
        try:
            import openai
        except ImportError as exc:
            raise EmbeddingProviderUnavailableError("openai package is not installed.", provider=self.name) from exc

        import os
        key = self.api_key or os.getenv("OPENAI_API_KEY") or "placeholder-key"
        kwargs = {"api_key": key, "timeout": self.timeout_seconds}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return openai.OpenAI(**kwargs)

    def embed_text(self, text: str) -> List[float]:
        client = self._get_client()
        try:
            resp = client.embeddings.create(input=text, model=self.model)
            if not resp.data:
                raise EmbeddingProviderError("No embedding data returned by OpenAI.", provider=self.name)
            return [float(x) for x in resp.data[0].embedding]
        except Exception as exc:
            err_msg = str(exc)
            if "RateLimitError" in type(exc).__name__ or "429" in err_msg:
                raise EmbeddingProviderRateLimitError(err_msg, provider=self.name) from exc
            if "APITimeoutError" in type(exc).__name__ or "timeout" in err_msg.lower():
                raise EmbeddingProviderTimeoutError(err_msg, provider=self.name) from exc
            raise EmbeddingProviderError(f"OpenAI embedding failed: {err_msg}", provider=self.name) from exc

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        client = self._get_client()
        try:
            resp = client.embeddings.create(input=texts, model=self.model)
            # Sort by index to preserve order
            sorted_data = sorted(resp.data, key=lambda d: d.index)
            return [[float(x) for x in item.embedding] for item in sorted_data]
        except Exception as exc:
            err_msg = str(exc)
            if "RateLimitError" in type(exc).__name__ or "429" in err_msg:
                raise EmbeddingProviderRateLimitError(err_msg, provider=self.name) from exc
            if "APITimeoutError" in type(exc).__name__ or "timeout" in err_msg.lower():
                raise EmbeddingProviderTimeoutError(err_msg, provider=self.name) from exc
            raise EmbeddingProviderError(f"OpenAI batch embedding failed: {err_msg}", provider=self.name) from exc
