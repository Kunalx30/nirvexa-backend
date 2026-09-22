"""
app/ai_engine/reasoning/task_executor.py
Task-Aware Model Execution Orchestrator for Nirvexa AI Engine.
Integrates ModelRouter, LocalModelDiscovery, LocalModelAdapter, and Cloud Providers.
Ensures zero automatic model downloads, strict no-substitution policy,
deterministic fallback handling, and robust credential protection.
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Type

from config import Config
from app.ai_engine.reasoning.model_router import (
    ModelRouter,
    ModelRouteDecision,
    SUPPORTED_TASKS,
    TARGET_LOCAL,
    TARGET_CLOUD,
    TARGET_VISION_LOCAL,
)
from app.ai_engine.reasoning.local_models import LocalModelDiscovery
from app.ai_engine.reasoning.local_adapter import (
    LocalModelAdapter,
    LocalModelResponse,
    ERR_LOCAL_RUNTIME_UNAVAILABLE,
    ERR_MODEL_NOT_CONFIGURED,
    ERR_MODEL_NOT_AVAILABLE,
    ERR_VISION_MODEL_NOT_CONFIGURED,
    ERR_VISION_EXECUTION_NOT_READY,
    ERR_LOCAL_MODEL_TIMEOUT,
    ERR_LOCAL_MODEL_EXECUTION_ERROR,
)
from app.ai_engine.reasoning.providers import (
    BaseLLMProvider,
    LLMProviderFactory,
    LLMProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)

# Standard Task Execution Error Codes
ERR_LOCAL_MODEL_UNAVAILABLE = "LOCAL_MODEL_UNAVAILABLE"
ERR_CLOUD_EXECUTION_ERROR = "CLOUD_EXECUTION_ERROR"
ERR_CLOUD_FALLBACK_FAILED = "CLOUD_FALLBACK_FAILED"
ERR_UNKNOWN_TASK = "UNKNOWN_TASK"


def _conf(key: str, default: Any) -> Any:
    """Safely retrieves configuration from Flask current_app or Config."""
    try:
        from flask import current_app
        if current_app and current_app.config:
            return current_app.config.get(key, default)
    except (ImportError, RuntimeError):
        pass
    return getattr(Config, key, default)


def _scrub_credentials(text: Optional[str]) -> Optional[str]:
    """Ensures raw credentials or sensitive bearer tokens are never exposed."""
    if not text:
        return text
    scrubbed = str(text)
    api_keys = [
        _conf("AI_ENGINE_LOCAL_API_KEY", ""),
        _conf("GEMINI_API_KEY", ""),
        _conf("AI_ENGINE_LLM_API_KEY", ""),
    ]
    for key in api_keys:
        if key and len(key) >= 6 and key in scrubbed:
            scrubbed = scrubbed.replace(key, "[REDACTED]")
    return scrubbed


@dataclass
class TaskExecutionResult:
    """
    Normalized result of executing a Nirvexa task.
    Encapsulates execution metrics, target attribution, and structured error codes.
    Guarantees no raw API keys or internal credentials are exposed.
    """
    success: bool
    task: str
    text: Optional[str] = None
    model: Optional[str] = None
    provider: str = "unknown"
    target: str = "unknown"
    latency_ms: float = 0.0
    tokens_used: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "task": self.task,
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "target": self.target,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "error_code": self.error_code,
            "error_message": _scrub_credentials(self.error_message),
            "metadata": dict(self.metadata),
        }


class TaskExecutor:
    """
    Task-aware model execution orchestrator.
    Integrates ModelRouter, LocalModelDiscovery, LocalModelAdapter, and Cloud Providers.
    Never downloads models or accepts arbitrary user-controlled endpoints.
    """

    def __init__(
        self,
        router: Optional[Type[ModelRouter]] = None,
        discovery: Optional[LocalModelDiscovery] = None,
        local_adapter: Optional[LocalModelAdapter] = None,
        cloud_provider: Optional[BaseLLMProvider] = None,
        policy: Optional[Any] = None,
    ):
        self.router = router or ModelRouter
        self.discovery = discovery or LocalModelDiscovery()
        self.local_adapter = local_adapter or LocalModelAdapter(discovery=self.discovery)
        self._cloud_provider = cloud_provider
        self.policy = policy

    def _execute_cloud(
        self,
        task: str,
        decision: ModelRouteDecision,
        prompt: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        metadata: Dict[str, Any],
        start_time: float,
    ) -> TaskExecutionResult:
        """Executes inference via the resolved cloud provider."""
        provider_name = decision.fallback_provider if metadata.get("fallback_triggered") else decision.provider
        model_name = decision.fallback_model if metadata.get("fallback_triggered") else decision.model

        try:
            if self._cloud_provider is not None:
                prov = self._cloud_provider
            else:
                prov = LLMProviderFactory.get_provider(provider_name, model=model_name)

            gen_result = prov.generate(
                prompt=prompt,
                system_instruction=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            elapsed = round((time.monotonic() - start_time) * 1000, 2)

            return TaskExecutionResult(
                success=True,
                task=task,
                text=gen_result.text,
                model=gen_result.model,
                provider=gen_result.provider,
                target=TARGET_CLOUD,
                latency_ms=elapsed,
                tokens_used=gen_result.tokens_used,
                metadata=metadata,
            )

        except (LLMProviderError, ProviderTimeoutError, ProviderUnavailableError, Exception) as exc:
            elapsed = round((time.monotonic() - start_time) * 1000, 2)
            error_code = ERR_CLOUD_FALLBACK_FAILED if metadata.get("fallback_triggered") else ERR_CLOUD_EXECUTION_ERROR
            logger.error("[TaskExecutor] Cloud execution failed for task=%s: %s", task, exc)
            return TaskExecutionResult(
                success=False,
                task=task,
                model=model_name,
                provider=provider_name,
                target=TARGET_CLOUD,
                latency_ms=elapsed,
                error_code=error_code,
                error_message=_scrub_credentials(str(exc)),
                metadata=metadata,
            )

    def execute(
        self,
        task: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        images: Optional[List[Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        allow_cloud_fallback: Optional[bool] = None,
        complexity: Optional[str] = None,
        **kwargs,
    ) -> TaskExecutionResult:
        """
        Executes a task by routing through ModelRouter, validating availability,
        and dispatching to LocalModelAdapter or Cloud provider with explicit fallback.
        """
        t0 = time.monotonic()

        # Security: Strip any user-supplied network endpoint to prevent SSRF
        if "base_url" in kwargs or "url" in kwargs or "provider_url" in kwargs:
            logger.warning("[TaskExecutor] User-supplied URL in kwargs discarded for SSRF prevention")
            kwargs.pop("base_url", None)
            kwargs.pop("url", None)
            kwargs.pop("provider_url", None)

        clean_task = str(task).strip().lower() if task else "text_generation"

        # Step 1: Query ModelRouter with discovery, policy, and complexity integration
        decision = self.router.resolve_route(
            clean_task,
            discovery=self.discovery,
            policy=self.policy,
            complexity=complexity,
        )

        # Fallback permission check
        can_fallback = (
            allow_cloud_fallback
            if allow_cloud_fallback is not None
            else bool(_conf("AI_ENGINE_LOCAL_FALLBACK_TO_CLOUD", True))
        )

        # =========================================================================
        # Target: LOCAL
        # =========================================================================
        if decision.target == TARGET_LOCAL:
            # 1. Local Runtime Unavailable
            if not self.discovery.is_runtime_available() or decision.status == "local_unavailable":
                elapsed = round((time.monotonic() - t0) * 1000, 2)
                if can_fallback and decision.fallback_target == TARGET_CLOUD:
                    logger.info("[TaskExecutor] Local runtime unavailable for task '%s' — falling back to cloud", clean_task)
                    meta = {
                        "fallback_triggered": True,
                        "original_target": TARGET_LOCAL,
                        "original_provider": decision.provider,
                        "original_model": decision.model,
                        "fallback_reason": ERR_LOCAL_RUNTIME_UNAVAILABLE,
                        "configured_model": decision.model,
                        "configured_model_available": False,
                        "recommended_model": decision.metadata.get("recommended_model"),
                        "policy_decision": decision.metadata.get("policy_decision"),
                    }
                    return self._execute_cloud(clean_task, decision, prompt, system_prompt, temperature, max_tokens, meta, t0)

                return TaskExecutionResult(
                    success=False,
                    task=clean_task,
                    model=decision.model,
                    provider=decision.provider,
                    target=TARGET_LOCAL,
                    latency_ms=elapsed,
                    error_code=ERR_LOCAL_RUNTIME_UNAVAILABLE,
                    error_message="Local model runtime is offline or unreachable.",
                    metadata={
                        "reason": "local_runtime_unavailable",
                        "configured_model": decision.model,
                        "configured_model_available": False,
                        "recommended_model": decision.metadata.get("recommended_model"),
                        "policy_decision": decision.metadata.get("policy_decision"),
                    },
                )

            # 2. Local Model Not Available / Not Installed (No silent substitution)
            if decision.status == "local_model_not_installed" or not self.discovery.is_model_available(decision.model):
                elapsed = round((time.monotonic() - t0) * 1000, 2)
                rec_model = decision.metadata.get("recommended_model")
                if can_fallback and decision.fallback_target == TARGET_CLOUD:
                    logger.info(
                        "[TaskExecutor] Local model '%s' not installed for task '%s' — falling back to cloud",
                        decision.model, clean_task,
                    )
                    meta = {
                        "fallback_triggered": True,
                        "original_target": TARGET_LOCAL,
                        "original_provider": decision.provider,
                        "original_model": decision.model,
                        "fallback_reason": ERR_LOCAL_MODEL_UNAVAILABLE,
                        "configured_model": decision.model,
                        "configured_model_available": False,
                        "recommended_model": rec_model,
                        "policy_decision": decision.metadata.get("policy_decision"),
                    }
                    return self._execute_cloud(clean_task, decision, prompt, system_prompt, temperature, max_tokens, meta, t0)

                return TaskExecutionResult(
                    success=False,
                    task=clean_task,
                    model=decision.model,
                    provider=decision.provider,
                    target=TARGET_LOCAL,
                    latency_ms=elapsed,
                    error_code=ERR_LOCAL_MODEL_UNAVAILABLE,
                    error_message=f"Configured local model '{decision.model}' is not available.",
                    metadata={
                        "reason": "local_model_not_installed",
                        "configured_model": decision.model,
                        "configured_model_available": False,
                        "recommended_model": rec_model,
                        "policy_decision": decision.metadata.get("policy_decision"),
                    },
                )

            # 3. Execute Local Model via LocalModelAdapter
            local_resp: LocalModelResponse = self.local_adapter.generate(
                prompt=prompt,
                model=decision.model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if local_resp.success:
                resp_meta = dict(local_resp.metadata)
                if "recommended_model" in decision.metadata:
                    resp_meta["recommended_model"] = decision.metadata["recommended_model"]
                if "policy_decision" in decision.metadata:
                    resp_meta["policy_decision"] = decision.metadata["policy_decision"]
                return TaskExecutionResult(
                    success=True,
                    task=clean_task,
                    text=local_resp.text,
                    model=local_resp.model,
                    provider=local_resp.provider,
                    target=TARGET_LOCAL,
                    latency_ms=local_resp.latency_ms,
                    tokens_used=local_resp.tokens_used,
                    metadata=resp_meta,
                )

            # Local execution failed: Check fallback eligibility
            if can_fallback and decision.fallback_target == TARGET_CLOUD:
                logger.info(
                    "[TaskExecutor] Local execution failed with error '%s' — falling back to cloud",
                    local_resp.error_code,
                )
                meta = {
                    "fallback_triggered": True,
                    "original_target": TARGET_LOCAL,
                    "original_provider": decision.provider,
                    "original_model": decision.model,
                    "original_error_code": local_resp.error_code,
                    "fallback_reason": local_resp.error_code,
                }
                return self._execute_cloud(clean_task, decision, prompt, system_prompt, temperature, max_tokens, meta, t0)

            # Normalize error code for caller
            err_code = (
                ERR_LOCAL_MODEL_UNAVAILABLE
                if local_resp.error_code == ERR_MODEL_NOT_AVAILABLE
                else local_resp.error_code
            )
            return TaskExecutionResult(
                success=False,
                task=clean_task,
                model=local_resp.model,
                provider=local_resp.provider,
                target=TARGET_LOCAL,
                latency_ms=local_resp.latency_ms,
                error_code=err_code,
                error_message=_scrub_credentials(local_resp.error_message),
                metadata=local_resp.metadata,
            )

        # =========================================================================
        # Target: VISION LOCAL
        # =========================================================================
        if decision.target == TARGET_VISION_LOCAL:
            elapsed = round((time.monotonic() - t0) * 1000, 2)

            # 1. Vision model unconfigured
            if decision.status == "vision_not_configured" or not decision.model:
                return TaskExecutionResult(
                    success=False,
                    task=clean_task,
                    target=TARGET_VISION_LOCAL,
                    latency_ms=elapsed,
                    error_code=ERR_VISION_MODEL_NOT_CONFIGURED,
                    error_message="Local vision model is not configured.",
                )

            # 2. Vision execution interface (delegated to LocalModelAdapter)
            vision_resp = self.local_adapter.generate_vision(
                prompt=prompt,
                images=images,
                model=decision.model,
                system_prompt=system_prompt,
                **kwargs,
            )

            err_code = (
                ERR_LOCAL_MODEL_UNAVAILABLE
                if vision_resp.error_code == ERR_MODEL_NOT_AVAILABLE
                else vision_resp.error_code
            )

            return TaskExecutionResult(
                success=False,
                task=clean_task,
                model=vision_resp.model,
                provider=vision_resp.provider,
                target=TARGET_VISION_LOCAL,
                latency_ms=vision_resp.latency_ms,
                error_code=err_code,
                error_message=_scrub_credentials(vision_resp.error_message),
                metadata=vision_resp.metadata,
            )

        # =========================================================================
        # Target: CLOUD
        # =========================================================================
        return self._execute_cloud(
            task=clean_task,
            decision=decision,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=decision.to_dict(),
            start_time=t0,
        )
