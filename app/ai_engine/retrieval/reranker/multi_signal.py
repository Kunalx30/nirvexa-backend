"""
app/ai_engine/retrieval/reranker/multi_signal.py
Advanced Multi-Signal Evidence Reranker combining:
- Lexical relevance (phrase match, token coverage, term density via KeywordReranker)
- Semantic relevance (pgvector cosine distance, RRF scores, or neutral proxy)
- Source quality (first-party document authority, curated DB, trusted TLD/domain heuristics)
- Freshness (exponential age decay based on publication/retrieval timestamps)
- Diversity / Redundancy penalty (MMR-style source concentration and token Jaccard penalty)

Fully deterministic, bounded O(N*K) complexity, zero external network calls,
and automatic graceful fallback to KeywordReranker on any failure.
"""
from datetime import datetime, timezone
import logging
import math
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from urllib.parse import urlparse

from app.ai_engine.retrieval.reranker.base import BaseReranker
from app.ai_engine.retrieval.reranker.keyword import KeywordReranker, STOP_WORDS
from app.ai_engine.retrieval.schemas import EvidenceItem

logger = logging.getLogger(__name__)

# Trusted authoritative domains and TLDs for conservative source quality scoring
_TRUSTED_TLDS = {".gov", ".edu", ".mil"}
_TRUSTED_AUTHORITY_DOMAINS = {
    "docs.python.org",
    "python.org",
    "kubernetes.io",
    "github.com",
    "stackoverflow.com",
    "wikipedia.org",
    "mozilla.org",
    "developer.mozilla.org",
    "microsoft.com",
    "learn.microsoft.com",
    "google.com",
    "cloud.google.com",
    "aws.amazon.com",
    "apache.org",
    "ietf.org",
    "w3.org",
    "arxiv.org",
    "nih.gov",
    "nist.gov",
}


def _get_config(key: str, default: Any) -> Any:
    try:
        from flask import current_app
        if current_app and current_app.config:
            return current_app.config.get(key, default)
    except (ImportError, RuntimeError):
        pass
    from config import Config
    return getattr(Config, key, default)


class MultiSignalReranker(BaseReranker):
    """
    Pluggable multi-signal evidence reranker.
    Evaluates evidence chunks combining lexical, semantic, source-quality, freshness,
    and diversity signals into a single normalized [0.0, 1.0] relevance score.
    """

    def __init__(
        self,
        weight_lexical: Optional[float] = None,
        weight_semantic: Optional[float] = None,
        weight_source_quality: Optional[float] = None,
        weight_freshness: Optional[float] = None,
        weight_diversity: Optional[float] = None,
        min_relevance_score: Optional[float] = None,
        filter_low_relevance: bool = True,
        keyword_reranker: Optional[KeywordReranker] = None,
    ):
        self.weight_lexical = (
            weight_lexical
            if weight_lexical is not None
            else _get_config("AI_ENGINE_RERANK_WEIGHT_LEXICAL", 0.35)
        )
        self.weight_semantic = (
            weight_semantic
            if weight_semantic is not None
            else _get_config("AI_ENGINE_RERANK_WEIGHT_SEMANTIC", 0.30)
        )
        self.weight_source_quality = (
            weight_source_quality
            if weight_source_quality is not None
            else _get_config("AI_ENGINE_RERANK_WEIGHT_SOURCE_QUALITY", 0.15)
        )
        self.weight_freshness = (
            weight_freshness
            if weight_freshness is not None
            else _get_config("AI_ENGINE_RERANK_WEIGHT_FRESHNESS", 0.10)
        )
        self.weight_diversity = (
            weight_diversity
            if weight_diversity is not None
            else _get_config("AI_ENGINE_RERANK_DIVERSITY_PENALTY", 0.10)
        )

        # Clamp weights safely to valid bounds
        self.weight_lexical = max(0.0, min(1.0, float(self.weight_lexical)))
        self.weight_semantic = max(0.0, min(1.0, float(self.weight_semantic)))
        self.weight_source_quality = max(0.0, min(1.0, float(self.weight_source_quality)))
        self.weight_freshness = max(0.0, min(1.0, float(self.weight_freshness)))
        self.weight_diversity = max(0.0, min(0.5, float(self.weight_diversity)))

        raw_min = (
            min_relevance_score
            if min_relevance_score is not None
            else _get_config("AI_ENGINE_MIN_RELEVANCE_SCORE", 0.10)
        )
        self.min_relevance_score = max(0.0, min(1.0, float(raw_min)))
        self.filter_low_relevance = filter_low_relevance

        # Keyword baseline reranker for lexical evaluation and zero-downtime fallback
        self.keyword_reranker = keyword_reranker or KeywordReranker(
            min_relevance_score=self.min_relevance_score,
            filter_low_relevance=False,  # We handle filtering in the multi-signal pass
        )
        self.fallback_reranker = KeywordReranker(
            min_relevance_score=self.min_relevance_score,
            filter_low_relevance=self.filter_low_relevance,
        )

    def rank(self, query: str, items: List[EvidenceItem]) -> List[EvidenceItem]:
        """
        Ranks and scores EvidenceItem candidates against query using multi-signal scoring.
        Gracefully falls back to KeywordReranker if any exception occurs.
        """
        if not items:
            return []

        query_clean = query.strip()
        if not query_clean:
            return items

        try:
            return self._rank_multi_signal(query_clean, items)
        except Exception as exc:
            logger.warning(
                "[MultiSignalReranker] Multi-signal ranking failed: %s (falling back to KeywordReranker)",
                exc,
            )
            return self.fallback_reranker.rank(query, items)

    def _rank_multi_signal(self, query: str, items: List[EvidenceItem]) -> List[EvidenceItem]:
        query_lower = query.lower()
        all_query_tokens = self.keyword_reranker._tokenize(query_lower)
        content_query_tokens = [t for t in all_query_tokens if t not in STOP_WORDS]
        active_tokens = content_query_tokens if content_query_tokens else all_query_tokens

        # Precompute individual signals for each item
        candidate_entries: List[Dict[str, Any]] = []
        token_cache: Dict[str, Set[str]] = {}

        w_sum = (
            self.weight_lexical
            + self.weight_semantic
            + self.weight_source_quality
            + self.weight_freshness
        )
        if w_sum <= 0.0:
            w_sum = 1.0

        for item in items:
            s_lex = self._compute_lexical_signal(query_lower, active_tokens, item)
            s_sem = self._compute_semantic_signal(s_lex, item)
            s_src = self._compute_source_quality_signal(item)
            s_fresh = self._compute_freshness_signal(item)

            # Weighted base score [0.0, 1.0]
            base_score = (
                self.weight_lexical * s_lex
                + self.weight_semantic * s_sem
                + self.weight_source_quality * s_src
                + self.weight_freshness * s_fresh
            ) / w_sum
            base_score = max(0.0, min(1.0, base_score))

            item_tokens = set(self.keyword_reranker._tokenize((item.text or "").lower()))
            token_cache[item.evidence_id] = item_tokens

            candidate_entries.append({
                "item": item,
                "base_score": base_score,
                "s_lex": s_lex,
                "s_sem": s_sem,
                "s_src": s_src,
                "s_fresh": s_fresh,
                "source_id": item.source_id,
                "tokens": item_tokens,
            })

        # Sequential greedy ranking with diversity / redundancy penalty (MMR style)
        selected_results: List[EvidenceItem] = []
        selected_source_counts: Dict[str, int] = {}
        remaining = list(candidate_entries)

        while remaining:
            best_idx = -1
            best_final_score = -1.0
            best_penalty = 0.0

            for idx, cand in enumerate(remaining):
                penalty = self._compute_diversity_penalty(
                    cand=cand,
                    selected_results=selected_results,
                    selected_source_counts=selected_source_counts,
                    token_cache=token_cache,
                )
                final_score = max(0.0, min(1.0, cand["base_score"] - penalty))

                # Deterministic selection with tie-breaking on text length and evidence_id
                item = cand["item"]
                if final_score > best_final_score:
                    best_final_score = final_score
                    best_idx = idx
                    best_penalty = penalty
                elif abs(final_score - best_final_score) < 1e-6 and best_idx >= 0:
                    curr_item = remaining[best_idx]["item"]
                    # Tie-break 1: longer informative text
                    if len(item.text) > len(curr_item.text):
                        best_idx = idx
                        best_final_score = final_score
                        best_penalty = penalty
                    elif len(item.text) == len(curr_item.text):
                        # Tie-break 2: lexicographical evidence_id
                        if item.evidence_id < curr_item.evidence_id:
                            best_idx = idx
                            best_final_score = final_score
                            best_penalty = penalty

            if best_idx < 0:
                break

            chosen = remaining.pop(best_idx)
            chosen_item = chosen["item"]
            rounded_final_score = round(best_final_score, 4)

            # Check low-relevance threshold
            if not self.filter_low_relevance or rounded_final_score >= self.min_relevance_score:
                chosen_item.relevance_score = rounded_final_score

                # Enrich metadata with explainable signal breakdown
                meta = dict(chosen_item.metadata or {})
                meta["rerank_signals"] = {
                    "lexical": round(chosen["s_lex"], 4),
                    "semantic": round(chosen["s_sem"], 4),
                    "source_quality": round(chosen["s_src"], 4),
                    "freshness": round(chosen["s_fresh"], 4),
                    "diversity_penalty": round(best_penalty, 4),
                    "base_score": round(chosen["base_score"], 4),
                    "final_score": rounded_final_score,
                }
                chosen_item.metadata = meta

                selected_results.append(chosen_item)
                src_id = chosen["source_id"]
                selected_source_counts[src_id] = selected_source_counts.get(src_id, 0) + 1

        # Final deterministic sort: (-relevance_score, -len(text), evidence_id)
        selected_results.sort(
            key=lambda x: (-x.relevance_score, -len(x.text), x.evidence_id)
        )
        return selected_results

    def _compute_lexical_signal(
        self, query_lower: str, active_tokens: List[str], item: EvidenceItem
    ) -> float:
        """Calculates lexical relevance score [0.0, 1.0] via KeywordReranker."""
        try:
            return self.keyword_reranker._compute_score(
                query_lower=query_lower,
                query_tokens=active_tokens,
                item=item,
            )
        except Exception:
            return 0.0

    def _compute_semantic_signal(self, lexical_score: float, item: EvidenceItem) -> float:
        """
        Extracts and normalizes semantic similarity [0.0, 1.0].
        Uses cosine_distance (from pgvector) or rrf_score (from RRF fusion),
        falling back gracefully to the lexical score if no vector score is present.
        """
        meta = item.metadata or {}

        # 1. Direct cosine distance from pgvector / in-memory vector search
        if "cosine_distance" in meta:
            try:
                dist = float(meta["cosine_distance"])
                # cosine_distance in [0.0, 2.0] where 0.0 is identical
                return max(0.0, min(1.0, 1.0 - dist))
            except (ValueError, TypeError):
                pass

        # 2. RRF score from Phase 5 fusion
        if "rrf_score" in meta:
            try:
                rrf = float(meta["rrf_score"])
                # Scale typical RRF range [0.01, 0.033] to [0.0, 1.0]
                return max(0.0, min(1.0, rrf * 30.0))
            except (ValueError, TypeError):
                pass

        # 3. If item matched in semantic retrieval mode
        if meta.get("retrieval_mode") == "semantic":
            return max(0.0, min(1.0, float(item.relevance_score or 0.75)))

        # Fallback for web search or un-embedded sources: proxy with lexical score
        return lexical_score

    def _compute_source_quality_signal(self, item: EvidenceItem) -> float:
        """
        Deterministic, conservative source quality score [0.0, 1.0].
        - Private tenant documents: 1.0 (authoritative first-party knowledge)
        - Curated Nirvexa database: 0.95
        - Authoritative TLDs (.gov, .edu): 0.85
        - Known high-reputation tech/docs domains: 0.90
        - Standard HTTPS web domains: 0.70
        - Unverified / HTTP: 0.50
        """
        src_type = item.source_type or "web"

        if src_type == "document":
            return 1.0
        if src_type == "nirvexa_db":
            return 0.95

        # For web results, evaluate domain reputation heuristics
        url_str = item.url or item.source_id or ""
        meta = item.metadata or {}
        domain = meta.get("domain") or ""

        if not domain and url_str:
            try:
                domain = urlparse(url_str).netloc.lower()
            except Exception:
                domain = ""

        domain = domain.lower().strip()
        if domain.startswith("www."):
            domain = domain[4:]

        if not domain:
            return 0.60

        # Check known high-authority domains
        if domain in _TRUSTED_AUTHORITY_DOMAINS:
            return 0.90
        for trusted_d in _TRUSTED_AUTHORITY_DOMAINS:
            if domain.endswith("." + trusted_d):
                return 0.90

        # Check authoritative TLDs (.gov, .edu, .mil)
        for tld in _TRUSTED_TLDS:
            if domain.endswith(tld):
                return 0.85

        # Standard HTTPS check
        if url_str.startswith("https://"):
            return 0.70

        return 0.50

    def _compute_freshness_signal(self, item: EvidenceItem) -> float:
        """
        Calculates temporal freshness score [0.0, 1.0] using exponential decay.
        If timestamps are missing or unparseable, falls back to a neutral 0.70.
        """
        meta = item.metadata or {}
        ts_str = (
            meta.get("published_at")
            or meta.get("retrieved_at")
            or meta.get("timestamp")
        )
        if not ts_str or not isinstance(ts_str, str):
            return 0.70

        try:
            # Normalize ISO string
            clean_ts = ts_str.strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            delta_days = max(0.0, (now - dt).total_seconds() / 86400.0)

            # Half-life lambda: ~346 days (1 year half-life)
            lam = 0.002
            freshness = math.exp(-lam * delta_days)
            return max(0.0, min(1.0, freshness))
        except (ValueError, TypeError, OverflowError):
            return 0.70

    def _compute_diversity_penalty(
        self,
        cand: Dict[str, Any],
        selected_results: List[EvidenceItem],
        selected_source_counts: Dict[str, int],
        token_cache: Dict[str, Set[str]],
    ) -> float:
        """
        Computes redundancy / diversity penalty based on:
        1. Concentration from the same source/document.
        2. Token Jaccard overlap with already selected items.
        Scaled by self.weight_diversity.
        """
        if not selected_results or self.weight_diversity <= 0.0:
            return 0.0

        # 1. Source concentration penalty
        src_id = cand["source_id"]
        count = selected_source_counts.get(src_id, 0)
        source_penalty = min(0.50, count * 0.25)

        # 2. Textual overlap penalty (Maximal Marginal Relevance)
        cand_tokens = cand["tokens"]
        max_jaccard = 0.0
        if cand_tokens:
            for chosen in selected_results:
                chosen_tokens = token_cache.get(chosen.evidence_id, set())
                if not chosen_tokens:
                    continue
                intersection = len(cand_tokens.intersection(chosen_tokens))
                union = len(cand_tokens.union(chosen_tokens))
                if union > 0:
                    jaccard = intersection / union
                    if jaccard > max_jaccard:
                        max_jaccard = jaccard

        text_penalty = max_jaccard * 0.60
        total_penalty = self.weight_diversity * (source_penalty + text_penalty)
        return max(0.0, min(0.50, total_penalty))
