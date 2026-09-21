"""
app/ai_engine/reasoning/__init__.py
Phase 3 AI Reasoning & Grounded Generation Layer package.
"""
from app.ai_engine.reasoning.schemas import (
    GroundingStatus,
    AnswerRequest,
    AnswerResponse,
    CitationItem,
    EvidenceSummary,
    GenerationMetadata,
    LLMGenerationResult,
)
from app.ai_engine.reasoning.prompt_builder import PromptBuilder, SYSTEM_GROUNDING_INSTRUCTION
from app.ai_engine.reasoning.validator import CitationValidator
from app.ai_engine.reasoning.engine import ReasoningEngine
from app.ai_engine.reasoning.providers import (
    BaseLLMProvider,
    LLMProviderFactory,
    LLMProviderError,
    ProviderUnavailableError,
    ProviderTimeoutError,
    ProviderRateLimitError,
    GeminiLLMProvider,
    OllamaLLMProvider,
    OpenAICompatibleProvider,
    MockLLMProvider,
)

__all__ = [
    "GroundingStatus",
    "AnswerRequest",
    "AnswerResponse",
    "CitationItem",
    "EvidenceSummary",
    "GenerationMetadata",
    "LLMGenerationResult",
    "PromptBuilder",
    "SYSTEM_GROUNDING_INSTRUCTION",
    "CitationValidator",
    "ReasoningEngine",
    "BaseLLMProvider",
    "LLMProviderFactory",
    "LLMProviderError",
    "ProviderUnavailableError",
    "ProviderTimeoutError",
    "ProviderRateLimitError",
    "GeminiLLMProvider",
    "OllamaLLMProvider",
    "OpenAICompatibleProvider",
    "MockLLMProvider",
]
