"""
app/ai_engine/reasoning/providers/gemini.py
Google Gemini Provider utilizing the pre-installed google.generativeai SDK.
"""
import os
import logging
from typing import Optional

from app.ai_engine.reasoning.providers.base import (
    BaseLLMProvider,
    LLMProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderRateLimitError,
)
from app.ai_engine.reasoning.schemas import LLMGenerationResult

logger = logging.getLogger(__name__)


class GeminiLLMProvider(BaseLLMProvider):
    """
    Google Gemini Provider for grounded synthesis.
    Uses existing platform credentials and configuration.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
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
        self.api_key = api_key
        self._configured = False

    @property
    def name(self) -> str:
        return "gemini"

    def _ensure_configured(self):
        if self._configured:
            return

        resolved_key = self.api_key
        if not resolved_key:
            try:
                from flask import current_app
                if current_app and current_app.config:
                    resolved_key = current_app.config.get("GEMINI_API_KEY")
            except (ImportError, RuntimeError):
                pass
        if not resolved_key:
            resolved_key = os.getenv("GEMINI_API_KEY")

        if not resolved_key:
            raise ProviderUnavailableError("GEMINI_API_KEY is not configured.", provider="gemini")

        import google.generativeai as genai
        genai.configure(api_key=resolved_key)
        self._configured = True

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMGenerationResult:
        self._ensure_configured()
        import google.generativeai as genai

        eff_temperature = temperature if temperature is not None else self.temperature
        eff_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        try:
            logger.info(
                "[GeminiProvider] Dispatching request model=%s max_tokens=%d timeout=%ds",
                self.model, eff_max_tokens, self.timeout_seconds,
            )
            model_instance = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_instruction,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=eff_max_tokens,
                    temperature=eff_temperature,
                ),
            )

            response = model_instance.generate_content(
                prompt,
                request_options={"timeout": self.timeout_seconds},
            )

            text = response.text if response and hasattr(response, "text") else ""
            if not text:
                raise LLMProviderError("Gemini returned empty or blocked response.", provider="gemini")

            return LLMGenerationResult(
                text=text.strip(),
                provider=self.name,
                model=self.model,
                tokens_used=None,
                raw_metadata={},
            )

        except (TimeoutError, Exception) as exc:
            err_msg = str(exc)
            if "deadline" in err_msg.lower() or "timeout" in err_msg.lower():
                raise ProviderTimeoutError(f"Gemini request timed out after {self.timeout_seconds}s.", provider="gemini") from exc
            if "quota" in err_msg.lower() or "rate" in err_msg.lower() or "429" in err_msg:
                raise ProviderRateLimitError(f"Gemini quota/rate limit exceeded: {exc}", provider="gemini") from exc
            if "key" in err_msg.lower() or "auth" in err_msg.lower() or "permission" in err_msg.lower():
                raise ProviderUnavailableError(f"Gemini authentication failure: {exc}", provider="gemini") from exc

            logger.error("[GeminiProvider] Generation failure: %s", exc)
            raise LLMProviderError(f"Gemini generation error: {exc}", provider="gemini") from exc
