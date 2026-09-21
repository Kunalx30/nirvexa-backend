"""
app/ai_engine/embeddings/mock.py
Deterministic zero-network embedding provider for unit testing and CI.
Generates unit-normalized float vectors using SHA-256 hash seeds.
"""
import hashlib
import math
from typing import List, Optional

from app.ai_engine.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingProviderTimeoutError,
    EmbeddingProviderRateLimitError,
)


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Mock embedding provider generating deterministic vectors without network requests.
    Supports injecting artificial errors to test fallback mechanisms.
    """

    def __init__(
        self,
        model: str = "mock-embedding",
        dimension: int = 768,
        timeout_seconds: int = 5,
        should_fail: bool = False,
        failure_type: Optional[str] = None,
    ):
        super().__init__(model=model, dimension=dimension, timeout_seconds=timeout_seconds)
        self.should_fail = should_fail
        self.failure_type = failure_type

    @property
    def name(self) -> str:
        return "mock"

    def _generate_vector(self, text: str) -> List[float]:
        """
        Creates a deterministic unit-normalized vector for given text.
        """
        if self.should_fail:
            if self.failure_type == "timeout":
                raise EmbeddingProviderTimeoutError("Mock provider request timed out.", provider=self.name)
            if self.failure_type == "rate_limit":
                raise EmbeddingProviderRateLimitError("Mock provider quota exceeded.", provider=self.name)
            raise EmbeddingProviderError("Mock provider generic failure.", provider=self.name)

        # Generate deterministic floats from SHA256 chunks
        raw_vals: List[float] = []
        seed = text.encode("utf-8")
        chunk_idx = 0

        while len(raw_vals) < self.dimension:
            h = hashlib.sha256(seed + chunk_idx.to_bytes(4, "big")).digest()
            chunk_idx += 1
            # Read 4-byte integers and convert to float [-1.0, 1.0]
            for i in range(0, len(h), 4):
                if len(raw_vals) >= self.dimension:
                    break
                int_val = int.from_bytes(h[i:i+4], "big", signed=True)
                raw_vals.append(int_val / 2147483648.0)

        # Normalize to unit length (L2 norm)
        norm = math.sqrt(sum(x * x for x in raw_vals))
        if norm == 0.0:
            norm = 1.0
        return [round(x / norm, 6) for x in raw_vals]

    def embed_text(self, text: str) -> List[float]:
        return self._generate_vector(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._generate_vector(t) for t in texts]
