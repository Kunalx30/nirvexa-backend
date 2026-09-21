"""
app/ai_engine/retrieval/schemas.py
Pydantic schemas and dataclasses for Phase 2 Retrieval and Evidence Layer.
Source-neutral abstractions supporting web, database, documents, and future knowledge sources.
"""
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator


SourceType = Literal["web", "nirvexa_db", "document", "knowledge", "unknown"]


class RetrievalResult(BaseModel):
    """
    Source-neutral representation of a candidate document retrieved from any provider
    (Web, Nirvexa Database, Private Documents, Knowledge Graph).
    """
    source_id: str = Field(..., description="Unique source identifier (e.g. canonical URL or record UUID).")
    source_type: SourceType = Field(default="web", description="Origin of the source data.")
    title: str = Field(default="", description="Source document or page title.")
    url: Optional[str] = Field(default=None, description="Source URL if applicable.")
    content: str = Field(default="", description="Full clean extracted body text.")
    snippet: Optional[str] = Field(default=None, description="Brief summary or search snippet.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary provenance and domain metadata.")
    retrieval_score: float = Field(default=0.0, description="Initial provider or retrieval ranking score.")
    relevance_score: float = Field(default=0.0, description="Post-reranking relevance score [0.0, 1.0].")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EvidenceItem(BaseModel):
    """
    Granular, verifiable evidence chunk strictly linked to a parent source.
    Never creates orphan evidence without traceable provenance.
    """
    evidence_id: str = Field(..., description="Deterministic unique identifier for this evidence chunk.")
    source_id: str = Field(..., description="Identifier of the originating source document.")
    text: str = Field(..., description="Extracted evidence text chunk.")
    title: str = Field(default="", description="Source title for downstream citations.")
    url: Optional[str] = Field(default=None, description="Source URL for citations.")
    source_type: SourceType = Field(default="web", description="Source origin type.")
    relevance_score: float = Field(default=0.0, description="Normalized relevance score [0.0, 1.0].")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata (index, character count, etc.).")


class SourceSummary(BaseModel):
    """
    High-level citation summary of an authoritative source contributing evidence.
    """
    source_id: str
    title: str
    url: Optional[str] = None
    source_type: str = "web"
    chunk_count: int = 0


class EvidencePack(BaseModel):
    """
    Complete, structured evidence package delivered to downstream consumers
    (e.g., future local LLM / SLM inference engine).
    """
    query: str
    items: List[EvidenceItem] = Field(default_factory=list)
    total_candidates: int = 0
    selected_items: int = 0
    total_characters: int = 0
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_summary: List[SourceSummary] = Field(default_factory=list)


class EvidenceRequest(BaseModel):
    """
    Input payload for requesting ranked evidence.
    """
    query: str = Field(..., description="The research or retrieval query.")
    max_results: int = Field(default=5, ge=1, le=10, description="Max search results to retrieve (1-10).")
    max_evidence_items: Optional[int] = Field(default=None, ge=1, le=50, description="Optional cap on evidence chunks.")
    max_evidence_chars: Optional[int] = Field(default=None, ge=100, le=100000, description="Optional cap on cumulative characters.")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Query must be a string.")
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Query cannot be empty or whitespace only.")
        if len(trimmed) < 2:
            raise ValueError("Query must be at least 2 characters long.")
        if len(trimmed) > 500:
            raise ValueError("Query must not exceed 500 characters.")
        return trimmed
