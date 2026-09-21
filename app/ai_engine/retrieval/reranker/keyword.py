"""
app/ai_engine/retrieval/reranker/keyword.py
Dependency-free, multi-signal keyword reranker.
Computes deterministic relevance scores based on query term overlap, exact phrases,
title relevance, snippet relevance, term frequency density, and content quality.
"""
import math
import re
from typing import List, Set, Optional

from app.ai_engine.retrieval.reranker.base import BaseReranker
from app.ai_engine.retrieval.schemas import EvidenceItem


# Common English stop words to deemphasize in token overlap
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "should", "shouldn't", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "wasn't", "we", "were", "weren't", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "won't", "would", "wouldn't", "you",
    "your", "yours", "yourself", "yourselves",
}


class KeywordReranker(BaseReranker):
    """
    Multi-signal baseline reranker.
    Evaluates evidence chunks against queries using:
    - Exact phrase matching (title and text)
    - Query token coverage (title, text, and snippet)
    - Term frequency density
    - Content quality and micro-chunk penalty
    """

    def __init__(self, min_relevance_score: float = 0.10, filter_low_relevance: bool = True):
        self.min_relevance_score = max(0.0, min(1.0, min_relevance_score))
        self.filter_low_relevance = filter_low_relevance

    def rank(self, query: str, items: List[EvidenceItem]) -> List[EvidenceItem]:
        """
        Calculates normalized relevance scores for items and returns them in
        stable descending order.
        """
        if not items:
            return []

        query_clean = query.strip()
        if not query_clean:
            return items

        query_lower = query_clean.lower()
        all_query_tokens = self._tokenize(query_lower)
        # Content tokens (exclude stop words if query has content words)
        content_query_tokens = [t for t in all_query_tokens if t not in STOP_WORDS]
        active_tokens = content_query_tokens if content_query_tokens else all_query_tokens

        scored_items: List[EvidenceItem] = []

        for item in items:
            score = self._compute_score(
                query_lower=query_lower,
                query_tokens=active_tokens,
                item=item,
            )
            # Update item's relevance score (rounded to 4 decimal places)
            item.relevance_score = round(score, 4)

            if not self.filter_low_relevance or item.relevance_score >= self.min_relevance_score:
                scored_items.append(item)

        # Deterministic, stable sort:
        # 1. Highest relevance score first (-relevance_score)
        # 2. Longer informative text tie-breaker (-len(text))
        # 3. Deterministic evidence_id tie-breaker
        scored_items.sort(
            key=lambda x: (-x.relevance_score, -len(x.text), x.evidence_id)
        )

        return scored_items

    def _compute_score(
        self,
        query_lower: str,
        query_tokens: List[str],
        item: EvidenceItem,
    ) -> float:
        """
        Computes normalized composite relevance score [0.0, 1.0].
        """
        if not query_tokens:
            return 0.0

        title_lower = (item.title or "").lower()
        text_lower = (item.text or "").lower()
        snippet_lower = str(item.metadata.get("snippet", "")).lower() if item.metadata else ""

        title_tokens = set(self._tokenize(title_lower))
        text_tokens = self._tokenize(text_lower)
        text_tokens_set = set(text_tokens)
        snippet_tokens = set(self._tokenize(snippet_lower))

        score = 0.0

        # Signal 1: Exact Phrase Match
        # If the multi-word query appears verbatim in the title or text
        is_multi_word = len(query_tokens) > 1
        if query_lower in title_lower:
            score += 0.35 if is_multi_word else 0.20
        elif query_lower in text_lower:
            score += 0.25 if is_multi_word else 0.15

        # Signal 2: Title Token Coverage (0.0 to 0.25)
        title_matches = sum(1 for t in query_tokens if t in title_tokens)
        title_coverage = title_matches / len(query_tokens)
        score += 0.25 * title_coverage

        # Signal 3: Text Token Coverage (0.0 to 0.30)
        text_matches = sum(1 for t in query_tokens if t in text_tokens_set)
        text_coverage = text_matches / len(query_tokens)
        score += 0.30 * text_coverage

        # Signal 4: Snippet Token Coverage (0.0 to 0.10)
        if snippet_tokens:
            snippet_matches = sum(1 for t in query_tokens if t in snippet_tokens)
            score += 0.10 * (snippet_matches / len(query_tokens))

        # Signal 5: Term Frequency (TF) Density in text (0.0 to 0.10)
        if text_tokens:
            total_tf = sum(text_tokens.count(t) for t in query_tokens)
            tf_density = total_tf / len(text_tokens)
            # Logarithmic scaling for term density to avoid saturation
            tf_score = min(0.10, math.log1p(tf_density * 10) * 0.05)
            score += tf_score

        # Signal 6: Micro-chunk quality penalty
        # Chunks shorter than 40 chars receive a discount
        if len(item.text) < 40:
            score *= 0.5

        # Clamp score strictly between 0.0 and 1.0
        return max(0.0, min(1.0, score))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric tokens."""
        return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
