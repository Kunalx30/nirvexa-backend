"""
app/ai_engine/retrieval/__init__.py
Phase 2 Retrieval and Evidence Layer package.
"""
from app.ai_engine.retrieval.schemas import (
    SourceType,
    RetrievalResult,
    EvidenceItem,
    EvidencePack,
    SourceSummary,
    EvidenceRequest,
)
from app.ai_engine.retrieval.chunking import TextChunk, TextChunker
from app.ai_engine.retrieval.reranker import BaseReranker, KeywordReranker
from app.ai_engine.retrieval.service import RetrievalService
from app.ai_engine.retrieval.evidence import EvidenceBuilder

__all__ = [
    "SourceType",
    "RetrievalResult",
    "EvidenceItem",
    "EvidencePack",
    "SourceSummary",
    "EvidenceRequest",
    "TextChunk",
    "TextChunker",
    "BaseReranker",
    "KeywordReranker",
    "RetrievalService",
    "EvidenceBuilder",
]
