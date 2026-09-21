"""
app/ai_engine/documents/service.py
Service for user document ingestion, chunking, database persistence,
strict multi-tenant authorization, and candidate retrieval for hybrid reasoning.
"""
import uuid
import logging
from typing import List, Optional, Dict, Any

from config import Config
from app.extensions import db
from app.models.ai_document import AIDocument, AIDocumentChunk
from app.ai_engine.documents.parser import DocumentParser, DocumentParserError
from app.ai_engine.retrieval.chunking import TextChunker
from app.ai_engine.retrieval.schemas import RetrievalResult

logger = logging.getLogger(__name__)


def _get_config(key: str, default: Any) -> Any:
    try:
        from flask import current_app
        if current_app and current_app.config:
            return current_app.config.get(key, default)
    except (ImportError, RuntimeError):
        pass
    return getattr(Config, key, default)


class QuotaExceededError(ValueError):
    """Raised when user exceeds document count or chunk capacity."""
    pass


class DocumentService:
    """
    Manages document storage, quotas, and user-isolated retrieval.
    """

    def __init__(self, chunker: Optional[TextChunker] = None):
        max_chunk = _get_config("AI_ENGINE_MAX_CHUNK_CHARS", 1000)
        overlap = _get_config("AI_ENGINE_CHUNK_OVERLAP_CHARS", 100)
        self.chunker = chunker or TextChunker(max_chunk_chars=max_chunk, overlap_chars=overlap)

    def upload_document(
        self,
        user_id: str,
        file_bytes: bytes,
        filename: str,
        title: Optional[str] = None,
    ) -> AIDocument:
        """
        Parses, chunks, and persists a user document into the database.
        Enforces size and quota limits.
        """
        max_size = _get_config("AI_ENGINE_MAX_DOC_SIZE_BYTES", 5242880)
        if len(file_bytes) > max_size:
            raise ValueError(f"File size ({len(file_bytes)} bytes) exceeds maximum limit of {max_size} bytes.")

        # Check document count quota
        max_docs = _get_config("AI_ENGINE_MAX_DOCS_PER_USER", 10)
        existing_doc_count = AIDocument.query.filter_by(user_id=user_id).count()
        if existing_doc_count >= max_docs:
            raise QuotaExceededError(f"User document limit reached ({max_docs} documents maximum).")

        # Parse document
        parsed = DocumentParser.parse(file_bytes, filename)
        doc_title = title.strip() if title and title.strip() else parsed.title

        # Chunk content
        chunks = self.chunker.chunk_text(parsed.text)
        if not chunks:
            raise DocumentParserError("No valid text chunks could be extracted from this document.")

        # Check chunk count quota
        max_chunks = _get_config("AI_ENGINE_MAX_DOC_CHUNKS_PER_USER", 250)
        existing_chunk_count = AIDocumentChunk.query.filter_by(user_id=user_id).count()
        if existing_chunk_count + len(chunks) > max_chunks:
            raise QuotaExceededError(
                f"Adding {len(chunks)} chunks would exceed your chunk quota of {max_chunks} chunks "
                f"({existing_chunk_count} currently used)."
            )

        # Create Document record
        doc_id = str(uuid.uuid4())
        doc_record = AIDocument(
            id=doc_id,
            user_id=user_id,
            filename=filename,
            title=doc_title,
            file_type=parsed.file_type,
            file_size=len(file_bytes),
            chunk_count=len(chunks),
        )
        db.session.add(doc_record)

        # Create Chunk records
        for ch in chunks:
            meta = dict(parsed.metadata)
            meta.update({
                "chunk_index": ch.chunk_index,
                "total_chunks": ch.total_chunks,
                "start_char": ch.start_char,
                "end_char": ch.end_char,
                "document_title": doc_title,
                "filename": filename,
            })
            chunk_record = AIDocumentChunk(
                id=f"chk_{uuid.uuid4().hex[:16]}",
                document_id=doc_id,
                user_id=user_id,
                chunk_index=ch.chunk_index,
                text=ch.text,
                char_count=ch.char_count,
                metadata_json=meta,
            )
            db.session.add(chunk_record)

        db.session.commit()
        logger.info(
            "[DocumentService] Uploaded doc_id=%s user_id=%s chunks=%d size=%d",
            doc_id, user_id, len(chunks), len(file_bytes),
        )
        return doc_record

    def list_documents(self, user_id: str) -> List[AIDocument]:
        """Returns all documents owned by user."""
        return (
            AIDocument.query.filter_by(user_id=user_id)
            .order_by(AIDocument.created_at.desc())
            .all()
        )

    def get_document(self, user_id: str, document_id: str) -> Optional[AIDocument]:
        """Returns a specific document owned by user or None."""
        return AIDocument.query.filter_by(id=document_id, user_id=user_id).first()

    def delete_document(self, user_id: str, document_id: str) -> bool:
        """
        Deletes a document and cascades deletion to all associated chunks.
        Strictly enforces tenant ownership.
        """
        doc = self.get_document(user_id, document_id)
        if not doc:
            return False

        AIDocumentChunk.query.filter_by(document_id=document_id).delete()
        db.session.delete(doc)
        db.session.commit()
        logger.info("[DocumentService] Deleted doc_id=%s for user_id=%s", document_id, user_id)
        return True

    def retrieve_candidates_for_query(
        self,
        user_id: str,
        query: str,
        max_candidates: int = 50,
    ) -> List[RetrievalResult]:
        """
        Retrieves all chunks belonging to user_id, converts them to RetrievalResults
        for downstream deduplication and KeywordReranker scoring.
        Strictly isolates chunks by user_id.
        """
        if not user_id:
            return []

        chunks = (
            AIDocumentChunk.query.filter_by(user_id=user_id)
            .limit(max_candidates)
            .all()
        )

        if not chunks:
            return []

        results: List[RetrievalResult] = []
        for idx, ch in enumerate(chunks):
            meta = dict(ch.metadata_json or {})
            title = meta.get("document_title") or f"Document {ch.document_id[:8]}"
            source_id = f"doc_{ch.document_id}"

            results.append(
                RetrievalResult(
                    source_id=source_id,
                    source_type="document",
                    title=title,
                    url=None,
                    content=ch.text,
                    snippet=ch.text[:200] if len(ch.text) > 200 else ch.text,
                    metadata=meta,
                    retrieval_score=round(1.0 / (idx + 1), 4),
                    relevance_score=0.0,
                    timestamp=ch.created_at.isoformat() if ch.created_at else "",
                )
            )

        return results
