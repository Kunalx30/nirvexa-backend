"""
app/ai_engine/reasoning/schemas.py
Pydantic schemas and dataclasses for Phase 3 AI Reasoning and Grounded Generation Layer.
"""
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator


GroundingStatus = Literal["grounded", "partially_grounded", "insufficient_evidence", "unsupported"]


class AnswerRequest(BaseModel):
    """
    Request payload for end-to-end grounded research and reasoning.
    """
    query: str = Field(..., description="The question or research query.")
    max_results: int = Field(default=5, ge=1, le=10, description="Max web search results to retrieve (1-10).")
    search_mode: Literal["web", "document", "hybrid"] = Field(default="web", description="Retrieval source mode.")
    max_evidence_items: Optional[int] = Field(default=None, ge=1, le=50, description="Max evidence chunks to evaluate.")
    max_evidence_chars: Optional[int] = Field(default=None, ge=100, le=100000, description="Max cumulative evidence characters.")

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


class CitationItem(BaseModel):
    """
    Authoritative citation strictly linked back to an evidence item.
    """
    citation_id: str = Field(..., description="Citation marker identifier (e.g. 'S1', 'S2').")
    source_id: str = Field(..., description="Identifier of the originating source document.")
    title: str = Field(default="", description="Source document or page title.")
    url: Optional[str] = Field(default=None, description="Source URL if available.")
    source_type: str = Field(default="web", description="Origin of the source (web, document, etc.).")
    snippet: Optional[str] = Field(default=None, description="Brief snippet or supporting sentence.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary provenance metadata.")


class GenerationMetadata(BaseModel):
    """
    Diagnostic and telemetry metadata for the generation step.
    """
    provider: str
    model: str
    execution_time_ms: float = 0.0
    retries_used: int = 0
    total_evidence_chunks: int = 0
    cited_chunks_count: int = 0


class EvidenceSummary(BaseModel):
    """
    Summary of the evidence consumed during reasoning.
    """
    total_candidates: int = 0
    evidence_items_used: int = 0
    total_characters: int = 0


class AnswerResponse(BaseModel):
    """
    Final, structured response combining grounded synthesis and verified citations.
    """
    query: str
    answer: str
    grounding_status: GroundingStatus
    citations: List[CitationItem] = Field(default_factory=list)
    evidence_summary: Optional[EvidenceSummary] = None
    generation_metadata: Optional[GenerationMetadata] = None
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LLMGenerationResult(BaseModel):
    """
    Normalized result returned by any BaseLLMProvider.
    """
    text: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)
