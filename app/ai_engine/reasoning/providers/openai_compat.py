"""
app/ai_engine/reasoning/providers/openai_compat.py
Generic OpenAI-Compatible Provider supporting DeepSeek, vLLM, LocalAI, or custom endpoints.
"""
import os
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


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    Generic provider for any OpenAI-compatible chat completions endpoint
    (vLLM, DeepSeek, LocalAI, Groq, or OpenAI).
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
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
        self.base_url = base_url
        self.api_key = api_key
        self._client: Optional[openai.OpenAI] = None

    @property
    def name(self) -> str:
        return "openai_compat"

    def _get_client(self) -> openai.OpenAI:
        if self._client is None:
            resolved_key = self.api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy-key"
            resolved_base_url = self.base_url or os.getenv("AI_ENGINE_OPENAI_BASE_URL") or "https://api.deepseek.com"

            self._client = openai.OpenAI(
                base_url=resolved_base_url,
                api_key=resolved_key,
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
                "[OpenAICompatibleProvider] Dispatching request model=%s max_tokens=%d",
                self.model, eff_max_tokens,
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=eff_max_tokens,
                temperature=eff_temperature,
            )

            if not response.choices or not response.choices[0].message:
                raise LLMProviderError("OpenAI-compatible endpoint returned empty choices.", provider="openai_compat")

            content = response.choices[0].message.content
            if content is None:
                raise LLMProviderError("OpenAI-compatible message content was None.", provider="openai_compat")

            tokens_used = getattr(response.usage, "total_tokens", None) if hasattr(response, "usage") else None

            return LLMGenerationResult(
                text=content.strip(),
                provider=self.name,
                model=self.model,
                tokens_used=tokens_used,
                raw_metadata={},
            )

        except openai.APITimeoutError as exc:
            raise ProviderTimeoutError(f"Request timed out after {self.timeout_seconds}s.", provider="openai_compat") from exc
        except openai.APIConnectionError as exc:
            raise ProviderUnavailableError(f"Could not connect to endpoint: {exc}", provider="openai_compat") from exc
        except openai.RateLimitError as exc:
            raise ProviderRateLimitError(f"Rate limit exceeded: {exc}", provider="openai_compat") from exc
        except Exception as exc:
            raise LLMProviderError(f"Generation failed: {exc}", provider="openai_compat") from exc
