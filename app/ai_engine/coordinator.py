"""
app/ai_engine/coordinator.py
High-level WebResearchEngine tying search, URL normalization, targeted fetching,
and content extraction into structured ResearchResponse objects.
"""
import time
from datetime import datetime, timezone
import logging
from typing import Optional, List, Tuple

from app.ai_engine.extraction.content_extractor import ContentExtractor
from app.ai_engine.fetch.url_validator import normalize_url
from app.ai_engine.fetch.web_fetcher import WebFetcher
from app.ai_engine.reasoning.engine import ReasoningEngine
from app.ai_engine.reasoning.schemas import AnswerResponse
from app.ai_engine.retrieval.evidence import EvidenceBuilder
from app.ai_engine.retrieval.schemas import EvidencePack, RetrievalResult
from app.ai_engine.schemas.research import (
    SearchRequest,
    SourceMetadata,
    WebSearchResult,
    ResearchResponse,
)
from app.ai_engine.search.models import RawSearchResult
from app.ai_engine.search.service import SearchService
from app.ai_engine.documents.service import DocumentService

logger = logging.getLogger(__name__)


class WebResearchEngine:
    """
    Phase 1, 2 & 3 Web Research, Evidence, and Grounded Reasoning Engine.
    Executes search, deduplicates URLs, safely fetches pages,
    extracts clean full content, constructs structured EvidencePacks,
    and synthesizes verifiable, grounded answers.
    """

    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        fetcher: Optional[WebFetcher] = None,
        evidence_builder: Optional[EvidenceBuilder] = None,
        reasoning_engine: Optional[ReasoningEngine] = None,
        document_service: Optional[DocumentService] = None,
    ):
        self.search_service = search_service or SearchService()
        self.fetcher = fetcher or WebFetcher()
        self.evidence_builder = evidence_builder or EvidenceBuilder()
        self.reasoning_engine = reasoning_engine or ReasoningEngine()
        self.document_service = document_service or DocumentService()

    def research(self, request: SearchRequest) -> ResearchResponse:
        """
        Execute end-to-end web research for a query.
        """
        query = request.query
        max_results = request.max_results
        retrieved_at = datetime.now(timezone.utc).isoformat()
        start_time = time.monotonic()

        logger.info(
            "[WebResearchEngine] Starting research query=%r max_results=%d provider=%s",
            query,
            max_results,
            self.search_service.provider.name,
        )

        # 1. Execute search query via provider (exceptions propagate to route handler)
        raw_results = self.search_service.execute_search(query=query, max_results=max_results)

        if not raw_results:
            elapsed = (time.monotonic() - start_time) * 1000
            logger.info(
                "[WebResearchEngine] Zero results from provider for query=%r elapsed_ms=%.0f",
                query, elapsed,
            )
            return ResearchResponse(
                query=query,
                results=[],
                total_results=0,
                retrieved_at=retrieved_at,
                search_status="no_results",
                status_message="",
            )

        logger.info("[WebResearchEngine] Provider returned %d raw results for query=%r", len(raw_results), query)

        # 2. Normalize and deduplicate URLs before fetching.
        # We compute canonical/normalized URLs once and map each unique canonical URL
        # to its raw result. This guarantees consistency across:
        # - deduplication
        # - fetching
        # - fetch_results keys
        # - coordinator lookup
        # while preserving the original raw URL in the API response.
        seen_canonical: set = set()
        deduped_entries: List[Tuple[RawSearchResult, str]] = []

        for raw in raw_results:
            if not raw.url:
                continue
            canonical_url = normalize_url(raw.url) or raw.url
            if canonical_url in seen_canonical:
                continue
            seen_canonical.add(canonical_url)
            deduped_entries.append((raw, canonical_url))
            if len(deduped_entries) >= max_results:
                break

        urls_to_fetch = [canonical_url for _, canonical_url in deduped_entries]

        # 3. Targeted page fetching (if requested)
        fetch_results = {}
        if request.fetch_content and urls_to_fetch:
            logger.info("[WebResearchEngine] Fetching %d pages for query=%r", len(urls_to_fetch), query)
            fetch_results = self.fetcher.fetch_many(urls_to_fetch)

            # Log fetch outcomes
            successful = sum(1 for r in fetch_results.values() if r.success)
            failed = len(fetch_results) - successful
            logger.info(
                "[WebResearchEngine] Fetch complete: %d succeeded, %d failed for query=%r",
                successful, failed, query,
            )

        # 4. Construct normalized WebSearchResult list
        final_results = []

        for raw, canonical_url in deduped_entries:
            content_text = ""
            metadata = None
            fetch_status = "skipped"
            fetch_error = None

            if request.fetch_content:
                # Direct lookup using canonical_url, with raw.url fallback for mock compatibility
                fetch_res = fetch_results.get(canonical_url) or fetch_results.get(raw.url)

                if fetch_res is not None:
                    if fetch_res.success and fetch_res.html:
                        try:
                            extracted = ContentExtractor.extract(fetch_res.html, fetch_res.final_url)
                            content_text = extracted.text
                            metadata = extracted.metadata
                            fetch_status = "fetched"
                            # Enrich title if search title was empty
                            if not raw.title and extracted.title:
                                raw.title = extracted.title
                            if not content_text:
                                fetch_status = "empty"
                        except Exception as exc:
                            logger.warning(
                                "[WebResearchEngine] Content extraction failed for %s: %s",
                                fetch_res.final_url, exc,
                            )
                            fetch_status = "error"
                            fetch_error = "Content extraction failed"
                    elif fetch_res.success and not fetch_res.html:
                        fetch_status = "empty"
                    else:
                        err = fetch_res.error or ""
                        if "SSRF" in err or "blocked" in err.lower():
                            fetch_status = "blocked"
                        else:
                            fetch_status = "error"
                        fetch_error = err

            # Fallback metadata if fetch failed or content wasn't fetched
            if metadata is None:
                metadata = SourceMetadata(
                    title=raw.title,
                    description=raw.snippet,
                    author=None,
                    published_at=raw.published_at,
                    canonical_url=raw.url,
                    domain=raw.source or "",
                    retrieved_at=retrieved_at,
                )

            # If content extraction yielded empty (or failed), fallback to snippet
            if not content_text:
                content_text = raw.snippet or ""

            result_item = WebSearchResult(
                title=raw.title or "Untitled",
                url=raw.url,  # Preserves the original raw URL in the API response
                snippet=raw.snippet or "",
                source=raw.source or metadata.domain or "web",
                published_at=raw.published_at,
                content=content_text,
                retrieved_at=retrieved_at,
                metadata=metadata,
                fetch_status=fetch_status,
                fetch_error=fetch_error,
            )
            final_results.append(result_item)

        elapsed = (time.monotonic() - start_time) * 1000
        fetched_count = sum(1 for r in final_results if r.fetch_status == "fetched")
        logger.info(
            "[WebResearchEngine] Done query=%r status=success results=%d fetched=%d elapsed_ms=%.0f",
            query, len(final_results), fetched_count, elapsed,
        )

        return ResearchResponse(
            query=query,
            results=final_results,
            total_results=len(final_results),
            retrieved_at=retrieved_at,
            search_status="success",
            status_message="",
        )

    def research_evidence(
        self,
        request: SearchRequest,
        max_evidence_items: Optional[int] = None,
        max_evidence_chars: Optional[int] = None,
        search_mode: str = "web",
        user_id: Optional[str] = None,
    ) -> EvidencePack:
        """
        Phase 2 & 4: Execute research and/or document retrieval to build ranked EvidencePack.
        Supports search_mode: 'web', 'document', or 'hybrid'.
        Preserves complete backward compatibility with Phase 1 & 2.
        """
        mode = (search_mode or "web").lower().strip()

        if mode == "document":
            doc_candidates = self.document_service.retrieve_candidates_for_query(
                user_id=user_id or "",
                query=request.query,
            )
            return self.evidence_builder.build_pack(
                query=request.query,
                candidates=doc_candidates,
                max_evidence_items=max_evidence_items,
                max_evidence_chars=max_evidence_chars,
            )

        if mode == "hybrid":
            research_response = self.research(request)
            web_candidates = self.evidence_builder.retrieval_service.from_research_response(research_response)
            doc_candidates = self.document_service.retrieve_candidates_for_query(
                user_id=user_id or "",
                query=request.query,
            )
            combined_candidates = web_candidates + doc_candidates
            return self.evidence_builder.build_pack(
                query=request.query,
                candidates=combined_candidates,
                max_evidence_items=max_evidence_items,
                max_evidence_chars=max_evidence_chars,
            )

        # Default: 'web'
        research_response = self.research(request)
        return self.evidence_builder.build_pack(
            query=request.query,
            candidates=research_response,
            max_evidence_items=max_evidence_items,
            max_evidence_chars=max_evidence_chars,
        )

    def research_and_answer(
        self,
        request: SearchRequest,
        max_evidence_items: Optional[int] = None,
        max_evidence_chars: Optional[int] = None,
        search_mode: str = "web",
        user_id: Optional[str] = None,
    ) -> AnswerResponse:
        """
        Phase 3 & 4: Execute research/document retrieval, build evidence, and synthesize grounded answer.
        Does not perform additional web scraping in the reasoning layer.
        """
        evidence_pack = self.research_evidence(
            request=request,
            max_evidence_items=max_evidence_items,
            max_evidence_chars=max_evidence_chars,
            search_mode=search_mode,
            user_id=user_id,
        )
        return self.reasoning_engine.answer(
            query=request.query,
            evidence_pack=evidence_pack,
        )
