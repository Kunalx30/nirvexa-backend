"""
app/ai_engine/reasoning/providers/__init__.py
LLM Providers and Configuration-Driven Provider Factory.
"""
from typing import Optional, Dict, Any

from config import Config
from app.ai_engine.reasoning.providers.base import (
    BaseLLMProvider,
    LLMProviderError,
    ProviderUnavailableError,
    ProviderTimeoutError,
    ProviderRateLimitError,
)
from app.ai_engine.reasoning.providers.gemini import GeminiLLMProvider
from app.ai_engine.reasoning.providers.ollama import OllamaLLMProvider
from app.ai_engine.reasoning.providers.openai_compat import OpenAICompatibleProvider
from app.ai_engine.reasoning.providers.mock import MockLLMProvider


class LLMProviderFactory:
    """
    Factory resolving BaseLLMProvider instances based on application configuration.
    """

    @classmethod
    def get_provider(
        cls,
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> BaseLLMProvider:
        from flask import current_app

        def _conf(key: str, default: Any) -> Any:
            try:
                if current_app and current_app.config:
                    return current_app.config.get(key, default)
            except (ImportError, RuntimeError):
                pass
            return getattr(Config, key, default)

        resolved_name = (
            provider_name or _conf("AI_ENGINE_LLM_PROVIDER", "gemini")
        ).lower().strip()

        resolved_timeout = (
            timeout_seconds if timeout_seconds is not None else _conf("AI_ENGINE_LLM_TIMEOUT_SECONDS", 30)
        )
        resolved_temp = (
            temperature if temperature is not None else _conf("AI_ENGINE_LLM_TEMPERATURE", 0.2)
        )
        resolved_tokens = (
            max_tokens if max_tokens is not None else _conf("AI_ENGINE_LLM_MAX_TOKENS", 1024)
        )

        if resolved_name == "gemini":
            resolved_model = model or _conf("AI_ENGINE_LLM_MODEL", "gemini-2.5-flash")
            return GeminiLLMProvider(
                model=resolved_model,
                timeout_seconds=resolved_timeout,
                temperature=resolved_temp,
                max_tokens=resolved_tokens,
                **kwargs,
            )

        if resolved_name == "ollama":
            resolved_model = model or _conf("AI_ENGINE_LLM_MODEL", "llama3.2")
            base_url = kwargs.get("base_url") or _conf("AI_ENGINE_OLLAMA_BASE_URL", "http://localhost:11434/v1")
            return OllamaLLMProvider(
                model=resolved_model,
                base_url=base_url,
                timeout_seconds=resolved_timeout,
                temperature=resolved_temp,
                max_tokens=resolved_tokens,
            )

        if resolved_name in ("openai_compat", "openai"):
            resolved_model = model or _conf("AI_ENGINE_LLM_MODEL", "deepseek-chat")
            return OpenAICompatibleProvider(
                model=resolved_model,
                timeout_seconds=resolved_timeout,
                temperature=resolved_temp,
                max_tokens=resolved_tokens,
                **kwargs,
            )

        if resolved_name == "mock":
            resolved_model = model or "mock-model"
            return MockLLMProvider(
                model=resolved_model,
                timeout_seconds=resolved_timeout,
                temperature=resolved_temp,
                max_tokens=resolved_tokens,
                **kwargs,
            )

        raise LLMProviderError(
            f"Unsupported AI Engine LLM provider: '{resolved_name}'. "
            f"Supported providers: gemini, ollama, openai_compat, mock.",
            provider=resolved_name,
        )


__all__ = [
    "BaseLLMProvider",
    "LLMProviderError",
    "ProviderUnavailableError",
    "ProviderTimeoutError",
    "ProviderRateLimitError",
    "GeminiLLMProvider",
    "OllamaLLMProvider",
    "OpenAICompatibleProvider",
    "MockLLMProvider",
    "LLMProviderFactory",
]
