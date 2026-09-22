"""
app/ai_engine/reasoning/engine.py
Core ReasoningEngine for coordinating grounded AI synthesis.
Consumes Phase 2 EvidencePack, builds injection-fenced prompts, manages provider
selection and retries, and validates inline citations without performing web requests.
"""
import time
import logging
from typing import Optional, Any

from config import Config
from app.ai_engine.reasoning.prompt_builder import PromptBuilder
from app.ai_engine.reasoning.providers import (
    BaseLLMProvider,
    LLMProviderFactory,
    LLMProviderError,
    ProviderTimeoutError,
    ProviderRateLimitError,
)
from app.ai_engine.reasoning.schemas import (
    AnswerResponse,
    EvidenceSummary,
    GenerationMetadata,
)
from app.ai_engine.reasoning.validator import CitationValidator
from app.ai_engine.retrieval.schemas import EvidencePack

logger = logging.getLogger(__name__)


def _get_config_value(key: str, default: any) -> any:
    try:
        from flask import current_app
        if current_app and current_app.config:
            return current_app.config.get(key, default)
    except (ImportError, RuntimeError):
        pass
    return getattr(Config, key, default)


class ReasoningEngine:
    """
    Coordinates grounded answer synthesis from an EvidencePack.
    Does NOT perform external web scraping.
    """

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        max_retries: Optional[int] = None,
        task_executor: Optional[Any] = None,
    ):
        self._provider = provider
        self.max_retries = (
            max_retries
            if max_retries is not None
            else _get_config_value("AI_ENGINE_LLM_MAX_RETRIES", 2)
        )
        self.task_executor = task_executor

    def _resolve_provider(self) -> BaseLLMProvider:
        if self._provider is not None:
            return self._provider
        return LLMProviderFactory.get_provider()

    def answer(
        self,
        query: str,
        evidence_pack: EvidencePack,
        provider: Optional[BaseLLMProvider] = None,
    ) -> AnswerResponse:
        """
        Synthesizes a grounded answer backed by the provided EvidencePack.
        Uses task-aware TaskExecutor routing.
        """
        start_time = time.monotonic()
        clean_query = query.strip()
        active_provider = provider or self._provider

        # Step 1: Handle empty EvidencePack without invoking LLM
        if not evidence_pack or not evidence_pack.items:
            logger.info("[ReasoningEngine] Empty EvidencePack for query=%r — skipping LLM call", clean_query)
            elapsed = (time.monotonic() - start_time) * 1000
            prov_name = active_provider.name if active_provider else "unknown"
            prov_model = active_provider.model if active_provider else "unknown"
            return AnswerResponse(
                query=clean_query,
                answer="Based on the retrieved sources, there is insufficient evidence to determine an answer to your query.",
                grounding_status="insufficient_evidence",
                citations=[],
                evidence_summary=EvidenceSummary(
                    total_candidates=evidence_pack.total_candidates if evidence_pack else 0,
                    evidence_items_used=0,
                    total_characters=0,
                ),
                generation_metadata=GenerationMetadata(
                    provider=prov_name,
                    model=prov_model,
                    execution_time_ms=round(elapsed, 2),
                    retries_used=0,
                    total_evidence_chunks=0,
                    cited_chunks_count=0,
                ),
            )

        # Step 2: Build fenced prompt and citation mapping
        sys_instruction, user_prompt, citation_map = PromptBuilder.build(clean_query, evidence_pack)

        # Step 3: Invoke TaskExecutor with retry logic for transient errors
        retries_used = 0
        exec_result = None
        last_error = None

        from app.ai_engine.reasoning.task_executor import TaskExecutor
        if self.task_executor is not None and provider is None:
            executor = self.task_executor
        else:
            executor = TaskExecutor(cloud_provider=active_provider)

        for attempt in range(self.max_retries + 1):
            retries_used = attempt
            try:
                exec_result = executor.execute(
                    task="research",
                    prompt=user_prompt,
                    system_prompt=sys_instruction,
                )
            except Exception as exc:
                last_error = exc
                is_retryable = attempt < self.max_retries
                if is_retryable:
                    sleep_sec = 0.5 * (2 ** attempt)
                    logger.warning(
                        "[ReasoningEngine] Retryable execution exception (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1, self.max_retries, exc, sleep_sec,
                    )
                    time.sleep(sleep_sec)
                    continue
                logger.error("[ReasoningEngine] Execution failed permanently on attempt %d: %s", attempt + 1, exc)
                raise

            if exec_result.success:
                break

            # Handle retryable failure codes from TaskExecutionResult
            err_code = exec_result.error_code or ""
            is_retryable = err_code in (
                "CLOUD_EXECUTION_ERROR",
                "LOCAL_MODEL_TIMEOUT",
                "PROVIDER_TIMEOUT",
                "PROVIDER_RATE_LIMIT",
            )
            if is_retryable and attempt < self.max_retries:
                sleep_sec = 0.5 * (2 ** attempt)
                logger.warning(
                    "[ReasoningEngine] Retryable error (attempt %d/%d) on task research: %s (%s). Retrying in %.1fs",
                    attempt + 1, self.max_retries, exec_result.error_code, exec_result.error_message, sleep_sec,
                )
                time.sleep(sleep_sec)
                continue

            break

        if exec_result is None or not exec_result.success:
            err_msg = exec_result.error_message if exec_result else "Generation returned no result."
            prov_name = exec_result.provider if exec_result else (active_provider.name if active_provider else "unknown")
            raise last_error or LLMProviderError(err_msg, provider=prov_name)

        # Step 4: Validate citations and evaluate grounding status
        grounding_status, valid_citations, hallucinated_ids = CitationValidator.validate(
            generated_text=exec_result.text or "",
            citation_map=citation_map,
            total_evidence_count=len(evidence_pack.items),
        )

        if hallucinated_ids:
            logger.warning(
                "[ReasoningEngine] Detected %d hallucinated citation(s): %s",
                len(hallucinated_ids), hallucinated_ids,
            )

        elapsed = (time.monotonic() - start_time) * 1000

        return AnswerResponse(
            query=clean_query,
            answer=exec_result.text or "",
            grounding_status=grounding_status,
            citations=valid_citations,
            evidence_summary=EvidenceSummary(
                total_candidates=evidence_pack.total_candidates,
                evidence_items_used=len(evidence_pack.items),
                total_characters=evidence_pack.total_characters,
            ),
            generation_metadata=GenerationMetadata(
                provider=exec_result.provider,
                model=exec_result.model or "unknown",
                execution_time_ms=round(elapsed, 2),
                retries_used=retries_used,
                total_evidence_chunks=len(evidence_pack.items),
                cited_chunks_count=len(valid_citations),
                target=exec_result.target,
                fallback_triggered=exec_result.metadata.get("fallback_triggered"),
                fallback_reason=exec_result.metadata.get("fallback_reason"),
            ),
        )
