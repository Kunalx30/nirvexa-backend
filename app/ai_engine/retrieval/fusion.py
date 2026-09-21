"""
app/ai_engine/retrieval/fusion.py
Reciprocal Rank Fusion (RRF) for combining multi-modal retrieval candidates
(lexical keyword search and semantic vector similarity search).
"""
import logging
from typing import List, Dict, Optional

from app.ai_engine.retrieval.schemas import RetrievalResult

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    lexical_candidates: List[RetrievalResult],
    semantic_candidates: List[RetrievalResult],
    k: int = 60,
    weight_lexical: float = 1.0,
    weight_semantic: float = 1.0,
    limit: Optional[int] = None,
) -> List[RetrievalResult]:
    """
    Combines ranked candidate lists using Reciprocal Rank Fusion:
        RRF_Score(item) = Σ [ weight_i / (k + rank_i(item)) ]
    where rank is 1-indexed.

    Preserves provenance and deduplicates items matching the same chunk or source.
    """
    scores: Dict[str, float] = {}
    item_map: Dict[str, RetrievalResult] = {}
    sources_matched: Dict[str, set] = {}

    # Process lexical candidates
    for rank, cand in enumerate(lexical_candidates, start=1):
        # Key on chunk id if present in metadata, or content hash / snippet
        key = cand.metadata.get("chunk_id") or cand.content[:100]
        score = weight_lexical / (k + rank)
        scores[key] = scores.get(key, 0.0) + score
        if key not in item_map:
            item_map[key] = cand
            sources_matched[key] = set()
        sources_matched[key].add("lexical")

    # Process semantic candidates
    for rank, cand in enumerate(semantic_candidates, start=1):
        key = cand.metadata.get("chunk_id") or cand.content[:100]
        score = weight_semantic / (k + rank)
        scores[key] = scores.get(key, 0.0) + score
        if key not in item_map:
            item_map[key] = cand
            sources_matched[key] = set()
        sources_matched[key].add("semantic")

    # Sort keys by descending fused RRF score
    sorted_keys = sorted(scores.keys(), key=lambda k_id: scores[k_id], reverse=True)

    if limit is not None and limit > 0:
        sorted_keys = sorted_keys[:limit]

    # Build final RetrievalResult list with updated scores and metadata
    fused_results: List[RetrievalResult] = []
    for key in sorted_keys:
        base_item = item_map[key]
        fused_score = round(scores[key], 6)
        meta = dict(base_item.metadata)
        meta["rrf_score"] = fused_score
        meta["match_types"] = sorted(list(sources_matched[key]))

        fused_cand = base_item.model_copy(
            update={
                "retrieval_score": fused_score,
                "metadata": meta,
            }
        )
        fused_results.append(fused_cand)

    logger.debug(
        "[RRF Fusion] Combined %d lexical and %d semantic candidates into %d fused results.",
        len(lexical_candidates),
        len(semantic_candidates),
        len(fused_results),
    )
    return fused_results
