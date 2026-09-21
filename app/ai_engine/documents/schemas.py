"""
app/ai_engine/documents/schemas.py
Pydantic schemas for Phase 4 Document Ingestion, Knowledge Management, and Hybrid Retrieval.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    """
    Summary representation of an ingested document.
    """
    document_id: str
    filename: str
    title: str
    file_type: str
    file_size: int
    chunk_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    """
    Response returned upon successful file upload and chunking.
    """
    document: DocumentSummary
    message: str = "Document uploaded and indexed successfully."


class DocumentListResponse(BaseModel):
    """
    List of documents owned by the authenticated user.
    """
    documents: List[DocumentSummary] = Field(default_factory=list)
    total_documents: int = 0
    total_chunks: int = 0


class ParsedDocument(BaseModel):
    """
    Internal representation of parsed document content before chunking.
    """
    text: str
    title: str
    file_type: str
    char_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
