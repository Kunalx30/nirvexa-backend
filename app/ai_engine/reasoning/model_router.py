"""
app/ai_engine/reasoning/model_router.py
Local/Cloud Model Router and Task Abstraction for the Nirvexa AI Engine.
Decouples task-specific model selection from provider execution, enabling future
routing across small local SLMs, local vision-language models, and cloud providers.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Set

from config import Config

logger = logging.getLogger(__name__)

# Supported Task Categories
TASK_TEXT_GENERATION = "text_generation"
TASK_RESUME = "resume"
TASK_CLASSIFICATION = "classification"
TASK_EXTRACTION = "extraction"
TASK_VISION = "vision"
TASK_DOCUMENT_VISION = "document_vision"
TASK_CHAT = "chat"
TASK_RESEARCH = "research"
TASK_SALARY_ANALYSIS = "salary_analysis"
TASK_COMPANY_REVIEW = "company_review"

SUPPORTED_TASKS: Set[str] = {
    TASK_TEXT_GENERATION,
    TASK_RESUME,
    TASK_CLASSIFICATION,
    TASK_EXTRACTION,
    TASK_VISION,
    TASK_DOCUMENT_VISION,
    TASK_CHAT,
    TASK_RESEARCH,
    TASK_SALARY_ANALYSIS,
    TASK_COMPANY_REVIEW,
}

# Supported Routing Targets
TARGET_LOCAL = "local"
TARGET_CLOUD = "cloud"
TARGET_VISION_LOCAL = "vision_local"

SUPPORTED_TARGETS: Set[str] = {
    TARGET_LOCAL,
    TARGET_CLOUD,
    TARGET_VISION_LOCAL,
}

# Task to Config Key Mapping
TASK_TARGET_CONFIG_KEYS: Dict[str, str] = {
    TASK_RESUME: "AI_ENGINE_RESUME_MODEL_TARGET",
    TASK_CLASSIFICATION: "AI_ENGINE_CLASSIFICATION_MODEL_TARGET",
    TASK_EXTRACTION: "AI_ENGINE_EXTRACTION_MODEL_TARGET",
    TASK_VISION: "AI_ENGINE_VISION_MODEL_TARGET",
    TASK_DOCUMENT_VISION: "AI_ENGINE_DOCUMENT_VISION_MODEL_TARGET",
    TASK_CHAT: "AI_ENGINE_CHAT_MODEL_TARGET",
    TASK_RESEARCH: "AI_ENGINE_RESEARCH_MODEL_TARGET",
    TASK_SALARY_ANALYSIS: "AI_ENGINE_SALARY_MODEL_TARGET",
    TASK_COMPANY_REVIEW: "AI_ENGINE_COMPANY_REVIEW_MODEL_TARGET",
    TASK_TEXT_GENERATION: "AI_ENGINE_TEXT_GENERATION_MODEL_TARGET",
}

# Initial Default Routing Policies
DEFAULT_TASK_TARGETS: Dict[str, str] = {
    TASK_RESUME: TARGET_LOCAL,
    TASK_CLASSIFICATION: TARGET_LOCAL,
    TASK_EXTRACTION: TARGET_LOCAL,
    TASK_VISION: TARGET_VISION_LOCAL,
    TASK_DOCUMENT_VISION: TARGET_VISION_LOCAL,
    TASK_CHAT: TARGET_CLOUD,
    TASK_RESEARCH: TARGET_CLOUD,
    TASK_SALARY_ANALYSIS: TARGET_CLOUD,
    TASK_COMPANY_REVIEW: TARGET_CLOUD,
    TASK_TEXT_GENERATION: TARGET_CLOUD,
}


@dataclass
class ModelRouteDecision:
    """
    Structured model resolution decision.
    Decouples WHAT handles a task from HOW the model executes.
    """
    task: str
    target: str
    provider: str
    model: Optional[str]
    status: str
    is_local: bool = False
    is_vision: bool = False
    fallback_target: Optional[str] = None
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "target": self.target,
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "is_local": self.is_local,
            "is_vision": self.is_vision,
            "fallback_target": self.fallback_target,
            "fallback_provider": self.fallback_provider,
            "fallback_model": self.fallback_model,
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


class ModelRouter:
    """
    Resolves execution targets, providers, and models based on task category
    and server-side configuration. Does not execute models or perform network calls.
    """

    @classmethod
    def is_routing_enabled(cls) -> bool:
        return bool(_conf("AI_ENGINE_MODEL_ROUTING_ENABLED", True))

    @classmethod
    def get_cloud_provider(cls) -> str:
        return str(_conf("AI_ENGINE_CLOUD_PROVIDER", _conf("AI_ENGINE_LLM_PROVIDER", "gemini"))).strip().lower()

    @classmethod
    def get_cloud_model(cls) -> str:
        return str(_conf("AI_ENGINE_LLM_MODEL", "gemini-2.5-flash")).strip()

    @classmethod
    def get_local_text_model(cls) -> str:
        return str(
            _conf("AI_ENGINE_LOCAL_TEXT_MODEL", _conf("AI_ENGINE_LOCAL_MODEL", "llama3.2"))
        ).strip()

    @classmethod
    def get_local_vision_model(cls) -> str:
        return str(_conf("AI_ENGINE_LOCAL_VISION_MODEL", "")).strip()

    @classmethod
    def resolve_route(
        cls,
        task: str,
        check_local_health: bool = False,
        local_provider: Optional[Any] = None,
        discovery: Optional[Any] = None,
        policy: Optional[Any] = None,
        complexity: Optional[str] = None,
    ) -> ModelRouteDecision:
        """
        Resolves the appropriate provider and model for a given task.

        Args:
            task: Task category string (e.g. 'resume', 'classification', 'vision', 'chat').
            check_local_health: If True, tests reachability of the local provider.
            local_provider: Optional pre-existing local provider instance for health checking.
            discovery: Optional LocalModelDiscovery instance to check model availability.
            policy: Optional LocalModelPolicy instance to assess suitability and recommendations.
            complexity: Optional complexity level string ('low', 'medium', 'high').

        Returns:
            ModelRouteDecision containing target, provider, model, and fallback indicators.
        """
        clean_task = str(task).strip().lower() if task else "text_generation"
        cloud_provider = cls.get_cloud_provider()
        cloud_model = cls.get_cloud_model()

        # Step 1: Check if routing feature flag is active
        if not cls.is_routing_enabled():
            logger.info("[ModelRouter] Routing disabled — routing task '%s' to default cloud provider", clean_task)
            return ModelRouteDecision(
                task=clean_task,
                target=TARGET_CLOUD,
                provider=cloud_provider,
                model=cloud_model,
                status="routing_disabled",
                is_local=False,
                is_vision=False,
            )

        # Step 2: Handle unknown tasks safely
        is_unknown_task = clean_task not in SUPPORTED_TASKS
        if is_unknown_task:
            logger.warning("[ModelRouter] Unknown task '%s' requested — falling back to cloud", clean_task)
            raw_target = TARGET_CLOUD
        else:
            config_key = TASK_TARGET_CONFIG_KEYS.get(clean_task)
            raw_target = _conf(config_key, DEFAULT_TASK_TARGETS.get(clean_task, TARGET_CLOUD))

        # Step 3: Validate and normalize target
        target = str(raw_target).strip().lower()
        if target not in SUPPORTED_TARGETS:
            logger.warning(
                "[ModelRouter] Invalid target '%s' configured for task '%s' — falling back to cloud",
                raw_target, clean_task,
            )
            return ModelRouteDecision(
                task=clean_task,
                target=TARGET_CLOUD,
                provider=cloud_provider,
                model=cloud_model,
                status="invalid_target_fallback",
                is_local=False,
                is_vision=False,
                fallback_target=TARGET_CLOUD,
                fallback_provider=cloud_provider,
                fallback_model=cloud_model,
                metadata={"invalid_target": raw_target},
            )

        # Step 4: Resolve based on target
        if target == TARGET_LOCAL:
            local_model = cls.get_local_text_model()
            status = "resolved" if not is_unknown_task else "unknown_task_fallback"

            if discovery is not None:
                try:
                    if not discovery.is_model_available(local_model):
                        status = "local_model_not_installed"
                except Exception as exc:
                    logger.warning("[ModelRouter] Discovery check failed: %s", type(exc).__name__)

            elif check_local_health:
                # Test connectivity without throwing
                try:
                    if local_provider is not None:
                        is_healthy = bool(local_provider.check_health())
                    else:
                        from app.ai_engine.reasoning.providers import LLMProviderFactory
                        probe_provider = LLMProviderFactory.get_provider("ollama", model=local_model)
                        is_healthy = bool(probe_provider.check_health())
                except Exception as exc:
                    logger.warning("[ModelRouter] Local health probe error: %s", type(exc).__name__)
                    is_healthy = False

                if not is_healthy:
                    status = "local_unavailable"

            decision = ModelRouteDecision(
                task=clean_task,
                target=TARGET_LOCAL,
                provider="ollama",
                model=local_model,
                status=status,
                is_local=True,
                is_vision=False,
                fallback_target=TARGET_CLOUD,
                fallback_provider=cloud_provider,
                fallback_model=cloud_model,
            )

        elif target == TARGET_VISION_LOCAL:
            vision_model = cls.get_local_vision_model()
            is_configured = bool(vision_model and vision_model.lower() not in ("not_configured", "none"))

            if not is_configured:
                logger.info(
                    "[ModelRouter] Local vision model not configured for task '%s'",
                    clean_task,
                )
                decision = ModelRouteDecision(
                    task=clean_task,
                    target=TARGET_VISION_LOCAL,
                    provider="ollama",
                    model=None,
                    status="vision_not_configured",
                    is_local=True,
                    is_vision=True,
                    fallback_target=TARGET_CLOUD,
                    fallback_provider=cloud_provider,
                    fallback_model=cloud_model,
                )
            else:
                status = "resolved"
                if discovery is not None:
                    try:
                        if not discovery.is_model_available(vision_model):
                            status = "vision_model_not_installed"
                    except Exception as exc:
                        logger.warning("[ModelRouter] Vision discovery check failed: %s", type(exc).__name__)

                decision = ModelRouteDecision(
                    task=clean_task,
                    target=TARGET_VISION_LOCAL,
                    provider="ollama",
                    model=vision_model,
                    status=status,
                    is_local=True,
                    is_vision=True,
                    fallback_target=TARGET_CLOUD,
                    fallback_provider=cloud_provider,
                    fallback_model=cloud_model,
                )

        else:
            # TARGET_CLOUD
            status = "resolved" if not is_unknown_task else "unknown_task_fallback"
            decision = ModelRouteDecision(
                task=clean_task,
                target=TARGET_CLOUD,
                provider=cloud_provider,
                model=cloud_model,
                status=status,
                is_local=False,
                is_vision=False,
            )

        # Advisory Policy Evaluation (Phase 8)
        active_policy = policy
        if active_policy is None and bool(_conf("AI_ENGINE_LOCAL_POLICY_ENABLED", False)):
            try:
                from app.ai_engine.reasoning.model_policy import LocalModelPolicy
                active_policy = LocalModelPolicy(discovery=discovery)
            except Exception as exc:
                logger.debug("[ModelRouter] Default policy instantiation skipped: %s", exc)

        if active_policy is not None:
            try:
                rec = active_policy.recommend_model(clean_task, complexity=complexity)
                if rec:
                    decision.metadata["recommended_model"] = rec
                if decision.model:
                    suit = active_policy.evaluate_suitability(clean_task, decision.model, complexity=complexity)
                    decision.metadata["model_suitable"] = suit.suitable
                    decision.metadata["policy_reasons"] = suit.reason_codes
                    decision.metadata["policy_decision"] = suit.to_dict()
                elif rec:
                    # Model not configured, evaluate recommended candidate for advisory metadata
                    sel = active_policy.select_model(clean_task, complexity=complexity)
                    decision.metadata["policy_decision"] = sel.to_dict()
            except Exception as exc:
                logger.debug("[ModelRouter] Policy evaluation skipped: %s", exc)

        return decision

    @classmethod
    def get_provider_for_task(cls, task: str, **kwargs) -> Any:
        """
        Resolves the task route and instantiates the appropriate BaseLLMProvider
        using LLMProviderFactory.

        Args:
            task: The task category name.
            **kwargs: Additional options forwarded to LLMProviderFactory.

        Returns:
            An instantiated BaseLLMProvider.

        Raises:
            LLMProviderError: If the resolved target is vision but no vision model is configured.
        """
        from app.ai_engine.reasoning.providers import LLMProviderFactory, LLMProviderError

        decision = cls.resolve_route(task)

        if decision.status == "vision_not_configured":
            raise LLMProviderError(
                f"Local vision model is not configured for task '{decision.task}'. "
                f"Set AI_ENGINE_LOCAL_VISION_MODEL in server configuration.",
                provider=decision.provider,
            )

        return LLMProviderFactory.get_provider(
            provider_name=decision.provider,
            model=decision.model,
            **kwargs,
        )
