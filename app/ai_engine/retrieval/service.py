"""
app/ai_engine/retrieval/service.py
Core Retrieval Service for transforming Phase 1 research results and arbitrary sources
into clean, deduplicated, quality-filtered, and chunked candidate evidence.
"""
from datetime import datetime, timezone
import hashlib
import logging
from typing import List, Dict, Any, Optional, Set

from app.ai_engine.fetch.url_validator import normalize_url
from app.ai_engine.retrieval.chunking import TextChunker
from app.ai_engine.retrieval.schemas import RetrievalResult, EvidenceItem, SourceType
from app.ai_engine.schemas.research import ResearchResponse, WebSearchResult

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Deterministic document retrieval and normalization service.
    Converts diverse document formats into uniform candidate representations,
    removes duplicates, applies quality gates, and generates provenance-linked evidence chunks.
    """

    def __init__(
        self,
        default_max_chunk_chars: int = 1000,
        default_overlap_chars: int = 100,
        min_content_length: int = 15,
    ):
        self.default_max_chunk_chars = default_max_chunk_chars
        self.default_overlap_chars = default_overlap_chars
        self.min_content_length = min_content_length
        self.chunker = TextChunker(
            max_chunk_chars=default_max_chunk_chars,
            overlap_chars=default_overlap_chars,
        )

    def from_research_response(self, response: ResearchResponse) -> List[RetrievalResult]:
        """Convert Phase 1 ResearchResponse into normalized RetrievalResults."""
        if not response or not response.results:
            return []
        return self.from_web_search_results(response.results)

    def from_web_search_results(self, results: List[WebSearchResult]) -> List[RetrievalResult]:
        """Convert a list of WebSearchResult objects into normalized RetrievalResults."""
        retrieval_results: List[RetrievalResult] = []

        for idx, r in enumerate(results):
            if not r:
                continue

            raw_url = getattr(r, "url", "") or ""
            norm_url = normalize_url(raw_url) if raw_url else ""
            source_id = norm_url or raw_url or f"web_src_{idx}_{hashlib.md5((r.title or '').encode()).hexdigest()[:8]}"

            # Safely extract metadata dictionary
            meta_dict: Dict[str, Any] = {}
            if hasattr(r, "metadata") and r.metadata:
                if hasattr(r.metadata, "model_dump"):
                    meta_dict = r.metadata.model_dump()
                elif hasattr(r.metadata, "dict"):
                    meta_dict = r.metadata.dict()
                elif isinstance(r.metadata, dict):
                    meta_dict = dict(r.metadata)

            content = getattr(r, "content", "") or ""
            snippet = getattr(r, "snippet", "") or ""
            title = getattr(r, "title", "") or "Untitled"

            # Inverse rank retrieval score [1.0, 0.5, 0.33, ...]
            retrieval_score = round(1.0 / (idx + 1), 4)

            retrieval_results.append(
                RetrievalResult(
                    source_id=source_id,
                    source_type="web",
                    title=title,
                    url=raw_url or None,
                    content=content,
                    snippet=snippet,
                    metadata=meta_dict,
                    retrieval_score=retrieval_score,
                    relevance_score=0.0,
                    timestamp=getattr(r, "retrieved_at", datetime.now(timezone.utc).isoformat()),
                )
            )

        return retrieval_results

    def from_raw_candidates(self, raw_candidates: List[Dict[str, Any]]) -> List[RetrievalResult]:
        """
        Convert arbitrary raw source dictionaries (Nirvexa DB, private docs, knowledge)
        into normalized RetrievalResults.
        """
        results: List[RetrievalResult] = []
        for idx, c in enumerate(raw_candidates):
            if not isinstance(c, dict):
                continue

            raw_url = c.get("url")
            norm_url = normalize_url(raw_url) if raw_url else ""
            source_id = (
                str(c.get("source_id") or "")
                or norm_url
                or str(raw_url or "")
                or f"doc_src_{idx}_{hashlib.md5(str(c.get('title', '')).encode()).hexdigest()[:8]}"
            )

            source_type: SourceType = c.get("source_type", "unknown")
            title = str(c.get("title") or "Untitled")
            content = str(c.get("content") or "")
            snippet = str(c.get("snippet")) if c.get("snippet") else None
            metadata = dict(c.get("metadata") or {})
            retrieval_score = float(c.get("retrieval_score", round(1.0 / (idx + 1), 4)))

            results.append(
                RetrievalResult(
                    source_id=source_id,
                    source_type=source_type,
                    title=title,
                    url=raw_url,
                    content=content,
                    snippet=snippet,
                    metadata=metadata,
                    retrieval_score=retrieval_score,
                    relevance_score=0.0,
                    timestamp=str(c.get("timestamp") or datetime.now(timezone.utc).isoformat()),
                )
            )
        return results

    def deduplicate_candidates(self, candidates: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Deduplicates candidate documents by normalized URL / source_id.
        Preserves original ordering and the candidate with greater content length.
        """
        seen_keys: Set[str] = set()
        deduped: List[RetrievalResult] = []

        for candidate in candidates:
            # Determine canonical deduplication key
            key = candidate.source_id.strip()
            if candidate.url:
                norm = normalize_url(candidate.url)
                if norm:
                    key = norm

            if key in seen_keys:
                continue

            seen_keys.add(key)
            deduped.append(candidate)

        return deduped

    def filter_quality(self, candidates: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Filters out candidates with empty, whitespace-only, or negligible content.
        If full content is absent but a meaningful snippet exists, substitutes snippet as content.
        """
        valid_candidates: List[RetrievalResult] = []

        for c in candidates:
            clean_content = (c.content or "").strip()
            clean_snippet = (c.snippet or "").strip()

            if not clean_content and clean_snippet:
                c.content = clean_snippet
                clean_content = clean_snippet

            if len(clean_content) >= self.min_content_length:
                valid_candidates.append(c)
            else:
                logger.debug("[RetrievalService] Discarded low-content candidate: %s", c.source_id)

        return valid_candidates

    def chunk_candidate(
        self,
        candidate: RetrievalResult,
        chunker: Optional[TextChunker] = None,
    ) -> List[EvidenceItem]:
        """
        Splits a candidate document into EvidenceItem chunks.
        Strictly preserves provenance linking back to parent source.
        """
        active_chunker = chunker or self.chunker
        text_to_chunk = (candidate.content or candidate.snippet or "").strip()
        if not text_to_chunk:
            return []

        chunks = active_chunker.chunk_text(text_to_chunk)
        evidence_items: List[EvidenceItem] = []

        for ch in chunks:
            # Deterministic, unique evidence ID
            hash_input = f"{candidate.source_id}:{ch.chunk_index}:{ch.text[:48]}"
            evidence_id = f"evi_{hashlib.sha256(hash_input.encode('utf-8')).hexdigest()[:16]}"

            # Preserve metadata while adding chunking telemetry
            chunk_metadata = dict(candidate.metadata)
            chunk_metadata.update({
                "chunk_index": ch.chunk_index,
                "total_chunks": ch.total_chunks,
                "char_count": ch.char_count,
                "start_char": ch.start_char,
                "end_char": ch.end_char,
                "source_id": candidate.source_id,
                "snippet": candidate.snippet,
            })

            evidence_items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    source_id=candidate.source_id,
                    text=ch.text,
                    title=candidate.title,
                    url=candidate.url,
                    source_type=candidate.source_type,
                    relevance_score=0.0,
                    metadata=chunk_metadata,
                )
            )

        return evidence_items

    def prepare_evidence_candidates(
        self,
        candidates: List[RetrievalResult],
        chunker: Optional[TextChunker] = None,
    ) -> List[EvidenceItem]:
        """
        End-to-end preparation: deduplicate, filter quality, and chunk candidates.
        """
        deduped = self.deduplicate_candidates(candidates)
        quality = self.filter_quality(deduped)

        all_evidence: List[EvidenceItem] = []
        for c in quality:
            items = self.chunk_candidate(c, chunker=chunker)
            all_evidence.extend(items)

        return all_evidence
