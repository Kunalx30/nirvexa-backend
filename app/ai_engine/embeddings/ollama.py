"""
app/ai_engine/embeddings/ollama.py
Ollama local embedding provider utilizing pre-installed httpx HTTP client.
Default model: nomic-embed-text (768 dimensions).
Connects to local or network Ollama daemon at http://localhost:11434.
"""
import logging
from typing import List, Optional
import httpx

from app.ai_engine.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingProviderTimeoutError,
    EmbeddingProviderUnavailableError,
)

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """
    Produces vector embeddings from a self-hosted Ollama instance.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        dimension: int = 768,
        timeout_seconds: int = 15,
    ):
        super().__init__(model=model, dimension=dimension, timeout_seconds=timeout_seconds)
        self.base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "ollama"

    def embed_text(self, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(url, json=payload)
                if resp.status_code != 200:
                    raise EmbeddingProviderError(
                        f"Ollama returned HTTP {resp.status_code}: {resp.text}", provider=self.name
                    )
                data = resp.json()
                emb = data.get("embedding")
                if not emb:
                    raise EmbeddingProviderError("No embedding returned by Ollama.", provider=self.name)
                return [float(x) for x in emb]
        except httpx.TimeoutException as exc:
            raise EmbeddingProviderTimeoutError(f"Ollama request timed out: {exc}", provider=self.name) from exc
        except httpx.ConnectError as exc:
            raise EmbeddingProviderUnavailableError(f"Could not connect to Ollama at {self.base_url}: {exc}", provider=self.name) from exc
        except Exception as exc:
            if isinstance(exc, EmbeddingProviderError):
                raise
            raise EmbeddingProviderError(f"Ollama embedding failed: {exc}", provider=self.name) from exc

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Ollama /api/embeddings processes one prompt at a time
        return [self.embed_text(t) for t in texts]
