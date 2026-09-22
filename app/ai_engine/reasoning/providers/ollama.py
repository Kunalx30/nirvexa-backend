"""
app/ai_engine/reasoning/providers/ollama.py
Ollama Local Model Provider utilizing the pre-installed standard OpenAI SDK.
Connects to local OpenAI-compatible endpoints (Ollama, LM Studio, vLLM, LocalAI)
with thread-safe bounded concurrency and lightweight health checking.
"""
import logging
import threading
from typing import Optional, Dict, Any

import openai

from app.ai_engine.reasoning.providers.base import (
    BaseLLMProvider,
    LLMProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderRateLimitError,
)
from app.ai_engine.reasoning.schemas import LLMGenerationResult

logger = logging.getLogger(__name__)

# Thread-safe global registry of concurrency semaphores keyed by base_url
_local_semaphore_lock = threading.Lock()
_local_semaphores: Dict[str, threading.Semaphore] = {}


def _get_local_semaphore(base_url: str, max_concurrency: int) -> threading.Semaphore:
    """
    Retrieves or creates a thread-safe Semaphore for the given local endpoint.
    Ensures bounded concurrency across concurrent Flask requests to prevent VRAM/RAM exhaustion.
    """
    clean_url = base_url.rstrip("/")
    with _local_semaphore_lock:
        key = f"{clean_url}::{max_concurrency}"
        if key not in _local_semaphores:
            _local_semaphores[key] = threading.Semaphore(max_concurrency)
        return _local_semaphores[key]


def _reset_local_semaphores() -> None:
    """Reset the registry for test isolation."""
    with _local_semaphore_lock:
        _local_semaphores.clear()


class OllamaLLMProvider(BaseLLMProvider):
    """
    Ollama & OpenAI-compatible Local Model Provider.
    Enables local SLM/LLM inference with zero new dependencies, resource-aware
    bounded concurrency, and non-blocking connectivity health checking.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        timeout_seconds: int = 60,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        max_concurrency: int = 1,
        health_check_timeout_seconds: int = 5,
        **kwargs,
    ):
        super().__init__(
            model=model,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "ollama"
        self.max_concurrency = max(1, min(int(max_concurrency), 8))
        self.health_check_timeout_seconds = max(1, min(int(health_check_timeout_seconds), 30))
        self._client: Optional[openai.OpenAI] = None

        # Allow passing an explicit semaphore for test injection, otherwise use registry
        self._semaphore = kwargs.get("semaphore") or _get_local_semaphore(
            self.base_url, self.max_concurrency
        )

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def semaphore(self) -> threading.Semaphore:
        return self._semaphore

    def _get_client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=float(self.timeout_seconds),
            )
        return self._client

    def check_health(self, timeout_seconds: Optional[int] = None) -> bool:
        """
        Lightweight connectivity check for the local model runtime.
        Returns True if reachable and responsive, False otherwise.
        Never raises exceptions or exposes credentials.
        """
        eff_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self.health_check_timeout_seconds
        )
        try:
            client = self._get_client()
            client.models.list(timeout=float(eff_timeout))
            return True
        except Exception as exc:
            logger.warning(
                "[OllamaProvider] Health check failed for local runtime at %s: %s",
                self.base_url,
                type(exc).__name__,
            )
            return False

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMGenerationResult:
        client = self._get_client()
        eff_temperature = temperature if temperature is not None else self.temperature
        eff_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        try:
            logger.info(
                "[OllamaProvider] Dispatching local request model=%s base_url=%s max_tokens=%d timeout=%ds",
                self.model, self.base_url, eff_max_tokens, self.timeout_seconds,
            )

            # Resource-conscious bounded concurrency: acquire semaphore slot during local inference
            with self._semaphore:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=eff_max_tokens,
                    temperature=eff_temperature,
                )

            if not response.choices or not response.choices[0].message:
                raise LLMProviderError("Ollama returned an empty response choice.", provider="ollama")

            content = response.choices[0].message.content
            if content is None:
                raise LLMProviderError("Ollama message content was None.", provider="ollama")

            tokens_used = getattr(response.usage, "total_tokens", None) if hasattr(response, "usage") else None

            return LLMGenerationResult(
                text=content.strip(),
                provider=self.name,
                model=self.model,
                tokens_used=tokens_used,
                raw_metadata={"base_url": self.base_url},
            )

        except openai.APITimeoutError as exc:
            logger.warning("[OllamaProvider] Request timed out: %s", exc)
            raise ProviderTimeoutError(f"Ollama request timed out after {self.timeout_seconds}s.", provider="ollama") from exc

        except openai.APIConnectionError as exc:
            logger.warning("[OllamaProvider] Connection failed to %s: %s", self.base_url, exc)
            raise ProviderUnavailableError(f"Could not connect to Ollama server at {self.base_url}.", provider="ollama") from exc

        except openai.RateLimitError as exc:
            logger.warning("[OllamaProvider] Rate limit / quota hit: %s", exc)
            raise ProviderRateLimitError("Ollama rate limit exceeded.", provider="ollama") from exc

        except openai.APIStatusError as exc:
            logger.error("[OllamaProvider] API status error: status=%s", exc.status_code)
            raise LLMProviderError(f"Ollama API returned HTTP error: {exc.status_code}", provider="ollama") from exc

        except Exception as exc:
            if isinstance(exc, LLMProviderError):
                raise
            logger.error("[OllamaProvider] Unexpected error during generation: %s", exc)
            raise LLMProviderError(f"Ollama generation failed: {exc}", provider="ollama") from exc
