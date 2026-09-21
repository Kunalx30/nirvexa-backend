"""
app/ai_engine/retrieval/evidence.py
Evidence Builder for assembling structured, provenance-verified, context-budgeted
EvidencePack objects from candidate documents.
"""
from datetime import datetime, timezone
import logging
from typing import List, Dict, Any, Optional, Union

from config import Config
from app.ai_engine.retrieval.chunking import TextChunker
from app.ai_engine.retrieval.reranker.base import BaseReranker
from app.ai_engine.retrieval.reranker.keyword import KeywordReranker
from app.ai_engine.retrieval.schemas import (
    RetrievalResult,
    EvidenceItem,
    EvidencePack,
    SourceSummary,
)
from app.ai_engine.retrieval.service import RetrievalService
from app.ai_engine.schemas.research import ResearchResponse, WebSearchResult

logger = logging.getLogger(__name__)


def _get_app_config(key: str, default: Any) -> Any:
    """Safely retrieves a configuration key from current_app or fallback."""
    try:
        from flask import current_app
        if current_app and current_app.config:
            return current_app.config.get(key, default)
    except (ImportError, RuntimeError):
        pass
    return getattr(Config, key, default)


class EvidenceBuilder:
    """
    Evidence Builder orchestrating candidate ingestion, chunking,
    relevance reranking, and context budget enforcement into an EvidencePack.
    """

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        reranker: Optional[BaseReranker] = None,
    ):
        self.retrieval_service = retrieval_service or RetrievalService(
            default_max_chunk_chars=_get_app_config("AI_ENGINE_MAX_CHUNK_CHARS", 1000),
            default_overlap_chars=_get_app_config("AI_ENGINE_CHUNK_OVERLAP_CHARS", 100),
        )
        self.reranker = reranker or KeywordReranker(
            min_relevance_score=_get_app_config("AI_ENGINE_MIN_RELEVANCE_SCORE", 0.10),
            filter_low_relevance=True,
        )

    def build_pack(
        self,
        query: str,
        candidates: Union[ResearchResponse, List[WebSearchResult], List[RetrievalResult], List[Dict[str, Any]]],
        max_evidence_items: Optional[int] = None,
        max_evidence_chars: Optional[int] = None,
        max_candidates: Optional[int] = None,
    ) -> EvidencePack:
        """
        Builds a complete, budgeted EvidencePack from any supported candidate source.
        """
        effective_max_items = (
            max_evidence_items
            if max_evidence_items is not None
            else _get_app_config("AI_ENGINE_MAX_EVIDENCE_ITEMS", 10)
        )
        effective_max_chars = (
            max_evidence_chars
            if max_evidence_chars is not None
            else _get_app_config("AI_ENGINE_MAX_EVIDENCE_CHARS", 12000)
        )
        effective_max_candidates = (
            max_candidates
            if max_candidates is not None
            else _get_app_config("AI_ENGINE_MAX_CANDIDATES", 20)
        )

        now_iso = datetime.now(timezone.utc).isoformat()

        # Step 1: Normalize inputs to List[RetrievalResult]
        normalized_candidates: List[RetrievalResult] = []
        if isinstance(candidates, ResearchResponse):
            normalized_candidates = self.retrieval_service.from_research_response(candidates)
        elif isinstance(candidates, list):
            if not candidates:
                normalized_candidates = []
            elif isinstance(candidates[0], WebSearchResult):
                normalized_candidates = self.retrieval_service.from_web_search_results(candidates)  # type: ignore
            elif isinstance(candidates[0], RetrievalResult):
                normalized_candidates = list(candidates)  # type: ignore
            elif isinstance(candidates[0], dict):
                normalized_candidates = self.retrieval_service.from_raw_candidates(candidates)  # type: ignore
            else:
                logger.warning("[EvidenceBuilder] Unrecognized candidate item type: %s", type(candidates[0]))
                normalized_candidates = []

        # Enforce max candidate documents
        if len(normalized_candidates) > effective_max_candidates:
            normalized_candidates = normalized_candidates[:effective_max_candidates]

        total_candidates_count = len(normalized_candidates)

        if not normalized_candidates:
            return EvidencePack(
                query=query,
                items=[],
                total_candidates=0,
                selected_items=0,
                total_characters=0,
                generated_at=now_iso,
                source_summary=[],
            )

        # Step 2: Deduplicate & quality filter candidates
        deduped = self.retrieval_service.deduplicate_candidates(normalized_candidates)
        quality = self.retrieval_service.filter_quality(deduped)

        if not quality:
            return EvidencePack(
                query=query,
                items=[],
                total_candidates=total_candidates_count,
                selected_items=0,
                total_characters=0,
                generated_at=now_iso,
                source_summary=[],
            )

        # Step 3: Chunk into raw EvidenceItems (strict provenance guarantee)
        raw_evidence: List[EvidenceItem] = []
        for cand in quality:
            chunks = self.retrieval_service.chunk_candidate(cand)
            raw_evidence.extend(chunks)

        if not raw_evidence:
            return EvidencePack(
                query=query,
                items=[],
                total_candidates=total_candidates_count,
                selected_items=0,
                total_characters=0,
                generated_at=now_iso,
                source_summary=[],
            )

        # Step 4: Relevance Reranking
        ranked_evidence = self.reranker.rank(query=query, items=raw_evidence)

        # Step 5: Enforce Context Budget
        selected_items: List[EvidenceItem] = []
        total_chars = 0

        for item in ranked_evidence:
            if len(selected_items) >= effective_max_items:
                break

            item_len = len(item.text)

            # Check character budget
            if total_chars + item_len > effective_max_chars:
                # If no item has been added yet and the first item exceeds budget,
                # truncate it safely so caller gets at least some evidence
                if not selected_items and effective_max_chars > 50:
                    truncated_text = item.text[:effective_max_chars]
                    meta = dict(item.metadata)
                    meta["truncated"] = True
                    truncated_item = item.model_copy(update={"text": truncated_text, "metadata": meta})
                    selected_items.append(truncated_item)
                    total_chars += len(truncated_text)
                break

            selected_items.append(item)
            total_chars += item_len

        # Step 6: Compile authoritative SourceSummary for citations
        source_counts: Dict[str, Dict[str, Any]] = {}
        for item in selected_items:
            sid = item.source_id
            if sid not in source_counts:
                source_counts[sid] = {
                    "source_id": sid,
                    "title": item.title,
                    "url": item.url,
                    "source_type": item.source_type,
                    "chunk_count": 0,
                }
            source_counts[sid]["chunk_count"] += 1

        summaries = [
            SourceSummary(
                source_id=data["source_id"],
                title=data["title"],
                url=data["url"],
                source_type=data["source_type"],
                chunk_count=data["chunk_count"],
            )
            for data in source_counts.values()
        ]

        return EvidencePack(
            query=query,
            items=selected_items,
            total_candidates=total_candidates_count,
            selected_items=len(selected_items),
            total_characters=total_chars,
            generated_at=now_iso,
            source_summary=summaries,
        )
