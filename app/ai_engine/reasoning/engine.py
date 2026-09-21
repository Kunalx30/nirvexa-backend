"""
app/ai_engine/reasoning/engine.py
Core ReasoningEngine for coordinating grounded AI synthesis.
Consumes Phase 2 EvidencePack, builds injection-fenced prompts, manages provider
selection and retries, and validates inline citations without performing web requests.
"""
import time
import logging
from typing import Optional

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
    ):
        self._provider = provider
        self.max_retries = (
            max_retries
            if max_retries is not None
            else _get_config_value("AI_ENGINE_LLM_MAX_RETRIES", 2)
        )

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
        """
        start_time = time.monotonic()
        clean_query = query.strip()
        active_provider = provider or self._resolve_provider()

        # Step 1: Handle empty EvidencePack without invoking LLM
        if not evidence_pack or not evidence_pack.items:
            logger.info("[ReasoningEngine] Empty EvidencePack for query=%r — skipping LLM call", clean_query)
            elapsed = (time.monotonic() - start_time) * 1000
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
                    provider=active_provider.name,
                    model=active_provider.model,
                    execution_time_ms=round(elapsed, 2),
                    retries_used=0,
                    total_evidence_chunks=0,
                    cited_chunks_count=0,
                ),
            )

        # Step 2: Build fenced prompt and citation mapping
        sys_instruction, user_prompt, citation_map = PromptBuilder.build(clean_query, evidence_pack)

        # Step 3: Invoke LLM provider with retry logic for transient errors
        retries_used = 0
        gen_result = None
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                gen_result = active_provider.generate(
                    prompt=user_prompt,
                    system_instruction=sys_instruction,
                )
                retries_used = attempt
                break
            except (ProviderTimeoutError, ProviderRateLimitError, LLMProviderError) as exc:
                last_error = exc
                retries_used = attempt
                is_retryable = getattr(exc, "retryable", False) or isinstance(exc, (ProviderTimeoutError, ProviderRateLimitError))
                if is_retryable and attempt < self.max_retries:
                    sleep_sec = 0.5 * (2 ** attempt)
                    logger.warning(
                        "[ReasoningEngine] Retryable error (attempt %d/%d) on provider %s: %s. Retrying in %.1fs",
                        attempt + 1, self.max_retries, active_provider.name, exc, sleep_sec,
                    )
                    time.sleep(sleep_sec)
                    continue
                logger.error("[ReasoningEngine] Provider failed permanently on attempt %d: %s", attempt + 1, exc)
                raise

        if gen_result is None:
            raise last_error or LLMProviderError("Generation returned no result.", provider=active_provider.name)

        # Step 4: Validate citations and evaluate grounding status
        grounding_status, valid_citations, hallucinated_ids = CitationValidator.validate(
            generated_text=gen_result.text,
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
            answer=gen_result.text,
            grounding_status=grounding_status,
            citations=valid_citations,
            evidence_summary=EvidenceSummary(
                total_candidates=evidence_pack.total_candidates,
                evidence_items_used=len(evidence_pack.items),
                total_characters=evidence_pack.total_characters,
            ),
            generation_metadata=GenerationMetadata(
                provider=gen_result.provider,
                model=gen_result.model,
                execution_time_ms=round(elapsed, 2),
                retries_used=retries_used,
                total_evidence_chunks=len(evidence_pack.items),
                cited_chunks_count=len(valid_citations),
            ),
        )
