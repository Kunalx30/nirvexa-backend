"""
app/ai_engine/reasoning/local_adapter.py
Local Model Execution Adapter for the Nirvexa AI Engine.
Bridges task routing (ModelRouter) and model discovery (LocalModelDiscovery)
with concrete provider execution (OllamaLLMProvider).
Never downloads models or fakes vision execution.
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from config import Config
from app.ai_engine.reasoning.local_models import LocalModelDiscovery
from app.ai_engine.reasoning.model_router import ModelRouteDecision
from app.ai_engine.reasoning.providers import (
    LLMProviderFactory,
    OllamaLLMProvider,
    LLMProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)

# Standard Local Execution Error Codes
ERR_LOCAL_RUNTIME_UNAVAILABLE = "LOCAL_RUNTIME_UNAVAILABLE"
ERR_MODEL_NOT_CONFIGURED = "MODEL_NOT_CONFIGURED"
ERR_MODEL_NOT_AVAILABLE = "MODEL_NOT_AVAILABLE"
ERR_VISION_MODEL_NOT_CONFIGURED = "VISION_MODEL_NOT_CONFIGURED"
ERR_VISION_EXECUTION_NOT_READY = "VISION_EXECUTION_NOT_READY"
ERR_LOCAL_MODEL_TIMEOUT = "LOCAL_MODEL_TIMEOUT"
ERR_LOCAL_MODEL_EXECUTION_ERROR = "LOCAL_MODEL_EXECUTION_ERROR"


@dataclass
class LocalModelResponse:
    """
    Structured execution response for local model invocations.
    Guarantees no raw API keys or internal credentials are exposed.
    """
    success: bool
    text: Optional[str] = None
    model: Optional[str] = None
    provider: str = "ollama"
    target: str = "local"
    latency_ms: float = 0.0
    tokens_used: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "target": self.target,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "metadata": dict(self.metadata),
        }


def _conf(key: str, default: Any) -> Any:
    """Safely retrieves configuration from Flask current_app or Config."""
    try:
        from flask import current_app
        if current_app and current_app.config:
            return current_app.config.get(key, default)
    except (ImportError, RuntimeError):
        pass
    return getattr(Config, key, default)


class LocalModelAdapter:
    """
    Adapter responsible for executing validated local model inference.
    Reuses OllamaLLMProvider and LocalModelDiscovery without competing concurrency layers.
    """

    def __init__(
        self,
        discovery: Optional[LocalModelDiscovery] = None,
        provider: Optional[OllamaLLMProvider] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ):
        self.discovery = discovery or LocalModelDiscovery(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        self._provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _get_provider(self, model: str) -> OllamaLLMProvider:
        """Resolves or instantiates an OllamaLLMProvider for the target model."""
        if self._provider is not None:
            if self._provider.model == model:
                return self._provider

        # Resolve via standard LLMProviderFactory to preserve single-point semaphore & settings
        kwargs = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.timeout_seconds:
            kwargs["timeout_seconds"] = self.timeout_seconds

        prov = LLMProviderFactory.get_provider("ollama", model=model, **kwargs)
        if isinstance(prov, OllamaLLMProvider):
            return prov
        return OllamaLLMProvider(model=model, **kwargs)

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LocalModelResponse:
        """
        Executes local text generation after validating runtime and model availability.
        Never substitutes another model if the configured model is missing.
        """
        t0 = time.monotonic()
        target_model = str(
            model
            or _conf("AI_ENGINE_LOCAL_TEXT_MODEL", _conf("AI_ENGINE_LOCAL_MODEL", "llama3.2"))
        ).strip()

        if not target_model:
            return LocalModelResponse(
                success=False,
                target="local",
                error_code=ERR_MODEL_NOT_CONFIGURED,
                error_message="No local model configured or supplied.",
            )

        # Step 1: Check runtime health
        if not self.discovery.is_runtime_available():
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            logger.warning("[LocalModelAdapter] Local runtime is unreachable for model=%s", target_model)
            return LocalModelResponse(
                success=False,
                model=target_model,
                target="local",
                latency_ms=elapsed,
                error_code=ERR_LOCAL_RUNTIME_UNAVAILABLE,
                error_message="Local model runtime is offline or unreachable.",
            )

        # Step 2: Check model installation (strictly no silent substitution)
        if not self.discovery.is_model_available(target_model):
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            logger.warning("[LocalModelAdapter] Model '%s' is not installed in local runtime", target_model)
            return LocalModelResponse(
                success=False,
                model=target_model,
                target="local",
                latency_ms=elapsed,
                error_code=ERR_MODEL_NOT_AVAILABLE,
                error_message=f"Model '{target_model}' is not installed in the local runtime.",
            )

        # Step 3: Execute inference through provider (with built-in semaphore protection)
        try:
            prov = self._get_provider(target_model)
            gen_result = prov.generate(
                prompt=prompt,
                system_instruction=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            elapsed = round((time.monotonic() - t0) * 1000, 2)

            return LocalModelResponse(
                success=True,
                text=gen_result.text,
                model=gen_result.model,
                provider=gen_result.provider,
                target="local",
                latency_ms=elapsed,
                tokens_used=gen_result.tokens_used,
            )

        except ProviderTimeoutError as exc:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            logger.warning("[LocalModelAdapter] Local model timed out: %s", exc)
            return LocalModelResponse(
                success=False,
                model=target_model,
                target="local",
                latency_ms=elapsed,
                error_code=ERR_LOCAL_MODEL_TIMEOUT,
                error_message=str(exc),
            )

        except ProviderUnavailableError as exc:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            logger.warning("[LocalModelAdapter] Local provider unavailable: %s", exc)
            return LocalModelResponse(
                success=False,
                model=target_model,
                target="local",
                latency_ms=elapsed,
                error_code=ERR_LOCAL_RUNTIME_UNAVAILABLE,
                error_message=str(exc),
            )

        except (LLMProviderError, Exception) as exc:
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            logger.error("[LocalModelAdapter] Local generation failed: %s", exc)
            return LocalModelResponse(
                success=False,
                model=target_model,
                target="local",
                latency_ms=elapsed,
                error_code=ERR_LOCAL_MODEL_EXECUTION_ERROR,
                error_message=str(exc),
            )

    def generate_vision(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LocalModelResponse:
        """
        Vision execution interface.
        Prepares the architectural contract for future vision execution.
        Returns explicit structured status codes; never fakes vision by calling text models.
        """
        t0 = time.monotonic()
        target_model = str(
            model or _conf("AI_ENGINE_LOCAL_VISION_MODEL", "")
        ).strip()

        # Step 1: Verify vision model configured
        if not target_model or target_model.lower() in ("not_configured", "none"):
            return LocalModelResponse(
                success=False,
                target="vision_local",
                error_code=ERR_VISION_MODEL_NOT_CONFIGURED,
                error_message="Local vision model is not configured.",
            )

        # Step 2: Check runtime availability
        if not self.discovery.is_runtime_available():
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            return LocalModelResponse(
                success=False,
                model=target_model,
                target="vision_local",
                latency_ms=elapsed,
                error_code=ERR_LOCAL_RUNTIME_UNAVAILABLE,
                error_message="Local model runtime is offline or unreachable.",
            )

        # Step 3: Check vision model installation
        if not self.discovery.is_model_available(target_model):
            elapsed = round((time.monotonic() - t0) * 1000, 2)
            return LocalModelResponse(
                success=False,
                model=target_model,
                target="vision_local",
                latency_ms=elapsed,
                error_code=ERR_MODEL_NOT_AVAILABLE,
                error_message=f"Vision model '{target_model}' is not installed in the local runtime.",
            )

        # Step 4: Vision execution boundary (not yet ready)
        elapsed = round((time.monotonic() - t0) * 1000, 2)
        return LocalModelResponse(
            success=False,
            model=target_model,
            target="vision_local",
            latency_ms=elapsed,
            error_code=ERR_VISION_EXECUTION_NOT_READY,
            error_message="Local vision execution is not ready; full multimodal inference will be implemented in a future phase.",
            metadata={"images_count": len(images) if images else 0},
        )

    def execute_route(
        self,
        decision: ModelRouteDecision,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LocalModelResponse:
        """
        Executes a resolved ModelRouteDecision using the appropriate local interface.
        """
        if decision.target == "local":
            return self.generate(
                prompt=prompt,
                model=decision.model,
                system_prompt=system_prompt,
                **kwargs,
            )

        if decision.target == "vision_local":
            return self.generate_vision(
                prompt=prompt,
                model=decision.model,
                system_prompt=system_prompt,
                **kwargs,
            )

        return LocalModelResponse(
            success=False,
            target=decision.target,
            provider=decision.provider,
            model=decision.model,
            error_code="UNSUPPORTED_LOCAL_TARGET",
            error_message=f"Target '{decision.target}' is not a local execution target.",
        )
