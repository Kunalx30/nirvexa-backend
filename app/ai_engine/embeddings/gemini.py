"""
app/ai_engine/embeddings/gemini.py
Google Gemini embedding provider utilizing pre-installed google-generativeai SDK.
Default model: text-embedding-004 (768 dimensions).
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


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """
    Produces dense vector embeddings using Google's text-embedding-004 model.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-004",
        dimension: int = 768,
        timeout_seconds: int = 15,
    ):
        super().__init__(model=model, dimension=dimension, timeout_seconds=timeout_seconds)
        self.api_key = api_key

    @property
    def name(self) -> str:
        return "gemini"

    def _get_sdk(self):
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise EmbeddingProviderUnavailableError(
                "google-generativeai package is not installed.", provider=self.name
            ) from exc

        if not self.api_key:
            import os
            self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise EmbeddingProviderError("Missing GEMINI_API_KEY.", provider=self.name)

        genai.configure(api_key=self.api_key)
        return genai

    def embed_text(self, text: str) -> List[float]:
        genai = self._get_sdk()
        model_name = self.model if self.model.startswith("models/") else f"models/{self.model}"
        try:
            result = genai.embed_content(
                model=model_name,
                content=text,
                task_type="retrieval_document",
            )
            emb = result.get("embedding", [])
            if not emb:
                raise EmbeddingProviderError("Empty embedding returned by Gemini.", provider=self.name)
            return [float(x) for x in emb]
        except Exception as exc:
            err_msg = str(exc)
            if "ResourceExhausted" in err_msg or "429" in err_msg:
                raise EmbeddingProviderRateLimitError(err_msg, provider=self.name) from exc
            if "DeadlineExceeded" in err_msg or "timeout" in err_msg.lower():
                raise EmbeddingProviderTimeoutError(err_msg, provider=self.name) from exc
            raise EmbeddingProviderError(f"Gemini embedding failed: {err_msg}", provider=self.name) from exc

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        genai = self._get_sdk()
        model_name = self.model if self.model.startswith("models/") else f"models/{self.model}"
        try:
            result = genai.embed_content(
                model=model_name,
                content=texts,
                task_type="retrieval_document",
            )
            embeddings = result.get("embedding", [])
            if len(embeddings) != len(texts):
                # Fallback to single calls if batch format differs
                return [self.embed_text(t) for t in texts]
            return [[float(x) for x in emb] for emb in embeddings]
        except Exception as exc:
            logger.warning("[GeminiEmbeddingProvider] Batch embed failed, falling back to sequential: %s", exc)
            return [self.embed_text(t) for t in texts]
