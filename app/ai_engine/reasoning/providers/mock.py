"""
app/ai_engine/reasoning/providers/mock.py
Mock LLM Provider for unit and integration testing with zero external network access.
"""
from typing import Optional, Callable, Any

from app.ai_engine.reasoning.providers.base import (
    BaseLLMProvider,
    LLMProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.ai_engine.reasoning.schemas import LLMGenerationResult


class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic mock provider for zero-network testing.
    Can be configured with static text, custom side-effects, or predefined responses.
    """

    def __init__(
        self,
        model: str = "mock-model",
        default_response: Optional[str] = None,
        timeout_seconds: int = 5,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        should_timeout: bool = False,
        should_be_unavailable: bool = False,
        should_fail: bool = False,
    ):
        super().__init__(
            model=model,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.default_response = (
            default_response
            if default_response is not None
            else (
                "FastAPI is a high-performance web framework based on Starlette and Pydantic [S1]. "
                "It provides automatic interactive API documentation and fast execution speeds [S2]."
            )
        )
        self.custom_handler: Optional[Callable[[str, Optional[str]], str]] = None
        self.should_timeout: bool = should_timeout
        self.should_fail: bool = should_fail
        self.should_be_unavailable: bool = should_be_unavailable
        self.call_count: int = 0
        self.last_prompt: Optional[str] = None
        self.last_system_instruction: Optional[str] = None

    @property
    def name(self) -> str:
        return "mock"

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMGenerationResult:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_instruction = system_instruction

        if self.should_timeout:
            raise ProviderTimeoutError("Mock provider request timed out.", provider="mock")

        if self.should_be_unavailable:
            raise ProviderUnavailableError("Mock provider service is unavailable.", provider="mock")

        if self.should_fail:
            raise LLMProviderError("Mock provider failed generation.", provider="mock")

        if self.custom_handler:
            response_text = self.custom_handler(prompt, system_instruction)
        else:
            response_text = self.default_response

        return LLMGenerationResult(
            text=response_text,
            provider=self.name,
            model=self.model,
            tokens_used=len(response_text.split()),
            raw_metadata={"mock": True, "call_count": self.call_count},
        )
