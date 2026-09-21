"""
app/ai_engine/reasoning/providers/ollama.py
Ollama Local Model Provider utilizing the pre-installed standard OpenAI SDK.
Connects to Ollama's local OpenAI-compatible endpoint (http://localhost:11434/v1).
"""
import logging
from typing import Optional

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


class OllamaLLMProvider(BaseLLMProvider):
    """
    Ollama Local Model Provider.
    Enables local SLM/LLM inference (e.g. LLaMA 3.2, Qwen, Mistral) with zero new dependencies.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434/v1",
        timeout_seconds: int = 30,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        super().__init__(
            model=model,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.base_url = base_url.rstrip("/")
        self._client: Optional[openai.OpenAI] = None

    @property
    def name(self) -> str:
        return "ollama"

    def _get_client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI(
                base_url=self.base_url,
                api_key="ollama",  # Dummy key required by OpenAI client format
                timeout=float(self.timeout_seconds),
            )
        return self._client

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
                "[OllamaProvider] Dispatching request model=%s base_url=%s max_tokens=%d",
                self.model, self.base_url, eff_max_tokens,
            )
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
            logger.error("[OllamaProvider] API status error: status=%s response=%s", exc.status_code, exc.response)
            raise LLMProviderError(f"Ollama API returned HTTP error: {exc.status_code}", provider="ollama") from exc

        except Exception as exc:
            logger.error("[OllamaProvider] Unexpected error during generation: %s", exc)
            raise LLMProviderError(f"Ollama generation failed: {exc}", provider="ollama") from exc
