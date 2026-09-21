"""
app/ai_engine/documents/service.py
Service for user document ingestion, chunking, database persistence,
strict multi-tenant authorization, and hybrid (lexical + semantic) candidate retrieval.
"""
import math
import uuid
import logging
from typing import List, Optional, Dict, Any

from config import Config
from app.extensions import db
from app.models.ai_document import AIDocument, AIDocumentChunk
from app.ai_engine.documents.parser import DocumentParser, DocumentParserError
from app.ai_engine.embeddings.base import BaseEmbeddingProvider
from app.ai_engine.embeddings.factory import EmbeddingProviderFactory
from app.ai_engine.retrieval.chunking import TextChunker
from app.ai_engine.retrieval.fusion import reciprocal_rank_fusion
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
    Manages document storage, quotas, and user-isolated hybrid retrieval.
    """

    def __init__(
        self,
        chunker: Optional[TextChunker] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
    ):
        max_chunk = _get_config("AI_ENGINE_MAX_CHUNK_CHARS", 1000)
        overlap = _get_config("AI_ENGINE_CHUNK_OVERLAP_CHARS", 100)
        self.chunker = chunker or TextChunker(max_chunk_chars=max_chunk, overlap_chars=overlap)
        self.embedding_provider = embedding_provider

    def get_embedding_provider(self) -> Optional[BaseEmbeddingProvider]:
        """Lazily initializes the embedding provider when semantic retrieval is enabled."""
        if self.embedding_provider is not None:
            return self.embedding_provider

        if _get_config("AI_ENGINE_SEMANTIC_ENABLED", False):
            try:
                self.embedding_provider = EmbeddingProviderFactory.create()
                return self.embedding_provider
            except Exception as e:
                logger.warning("[DocumentService] Failed to initialize embedding provider: %s", e)
        return None

    def upload_document(
        self,
        user_id: str,
        file_bytes: bytes,
        filename: str,
        title: Optional[str] = None,
    ) -> AIDocument:
        """
        Parses, chunks, embeds (if enabled), and persists a user document into the database.
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

        # Generate vector embeddings if provider is available
        provider = self.get_embedding_provider()
        embeddings: List[Optional[List[float]]] = [None] * len(chunks)
        if provider:
            try:
                chunk_texts = [ch.text for ch in chunks]
                embeddings = provider.embed_batch(chunk_texts)
            except Exception as exc:
                logger.warning(
                    "[DocumentService] Embedding generation failed for %r: %s (falling back to None)",
                    filename, exc,
                )
                embeddings = [None] * len(chunks)

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
        for idx, ch in enumerate(chunks):
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
                embedding=embeddings[idx] if idx < len(embeddings) else None,
            )
            db.session.add(chunk_record)

        db.session.commit()
        logger.info(
            "[DocumentService] Uploaded doc_id=%s user_id=%s chunks=%d embedded=%d size=%d",
            doc_id, user_id, len(chunks), sum(1 for e in embeddings if e is not None), len(file_bytes),
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

    @staticmethod
    def _cosine_distance(vec_a: List[float], vec_b: List[float]) -> float:
        """Computes cosine distance [0.0, 2.0] between two vectors."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 1.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 1.0
        sim = dot / (norm_a * norm_b)
        return round(max(0.0, 1.0 - sim), 6)

    def _retrieve_semantic_candidates(
        self,
        user_id: str,
        query_vec: List[float],
        top_k: int = 25,
    ) -> List[RetrievalResult]:
        """
        Performs vector similarity search strictly scoped to user_id.
        Uses PostgreSQL pgvector `<=>` distance when on PostgreSQL, or
        gracefully calculates in-memory cosine distance for SQLite / fallback.
        """
        if not user_id or not query_vec:
            return []

        # Attempt PostgreSQL native pgvector query if connected to PostgreSQL
        try:
            if hasattr(db, "engine") and db.engine.dialect.name == "postgresql":
                from sqlalchemy import text
                vector_str = f"[{','.join(str(float(x)) for x in query_vec)}]"
                sql = text("""
                    SELECT id, document_id, text, metadata_json, created_at,
                           embedding <=> CAST(:query_vec AS vector) AS distance
                    FROM ai_document_chunks
                    WHERE user_id = :user_id AND embedding IS NOT NULL
                    ORDER BY distance ASC
                    LIMIT :limit
                """)
                rows = db.session.execute(sql, {
                    "user_id": user_id,
                    "query_vec": vector_str,
                    "limit": top_k,
                }).fetchall()

                results: List[RetrievalResult] = []
                for row in rows:
                    ch_id, doc_id, text_content, meta_json, created_at, dist = (
                        row[0], row[1], row[2], row[3], row[4], float(row[5])
                    )
                    meta = dict(meta_json or {})
                    meta["chunk_id"] = ch_id
                    meta["cosine_distance"] = dist
                    meta["retrieval_mode"] = "semantic"
                    title = meta.get("document_title") or f"Document {doc_id[:8]}"
                    sim_score = round(max(0.0, 1.0 - dist), 4)

                    results.append(
                        RetrievalResult(
                            source_id=f"doc_{doc_id}",
                            source_type="document",
                            title=title,
                            url=None,
                            content=text_content,
                            snippet=text_content[:200] if len(text_content) > 200 else text_content,
                            metadata=meta,
                            retrieval_score=sim_score,
                            relevance_score=0.0,
                            timestamp=created_at.isoformat() if created_at else "",
                        )
                    )
                return results

        except Exception as exc:
            logger.warning("[DocumentService] Native pgvector query failed, using fallback: %s", exc)

        # Fallback / SQLite path: load user's embedded chunks and rank with cosine distance
        chunks = (
            AIDocumentChunk.query.filter_by(user_id=user_id)
            .filter(AIDocumentChunk.embedding.isnot(None))
            .all()
        )
        if not chunks:
            return []

        scored_chunks = []
        for ch in chunks:
            emb = ch.embedding
            if isinstance(emb, (list, tuple)) and len(emb) == len(query_vec):
                dist = self._cosine_distance(query_vec, list(emb))
                scored_chunks.append((ch, dist))

        scored_chunks.sort(key=lambda x: x[1])
        top_chunks = scored_chunks[:top_k]

        results: List[RetrievalResult] = []
        for ch, dist in top_chunks:
            meta = dict(ch.metadata_json or {})
            meta["chunk_id"] = ch.id
            meta["cosine_distance"] = dist
            meta["retrieval_mode"] = "semantic"
            title = meta.get("document_title") or f"Document {ch.document_id[:8]}"
            sim_score = round(max(0.0, 1.0 - dist), 4)

            results.append(
                RetrievalResult(
                    source_id=f"doc_{ch.document_id}",
                    source_type="document",
                    title=title,
                    url=None,
                    content=ch.text,
                    snippet=ch.text[:200] if len(ch.text) > 200 else ch.text,
                    metadata=meta,
                    retrieval_score=sim_score,
                    relevance_score=0.0,
                    timestamp=ch.created_at.isoformat() if ch.created_at else "",
                )
            )
        return results

    def _retrieve_lexical_candidates(
        self,
        user_id: str,
        query: str,
        max_candidates: int = 50,
    ) -> List[RetrievalResult]:
        """
        Retrieves chunks belonging to user_id for lexical scoring.
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
            meta["chunk_id"] = ch.id
            meta["retrieval_mode"] = "lexical"
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

    def retrieve_candidates_for_query(
        self,
        user_id: str,
        query: str,
        max_candidates: int = 50,
    ) -> List[RetrievalResult]:
        """
        Retrieves candidate document chunks for user_id.
        If semantic search is enabled, executes hybrid retrieval:
            lexical candidates + semantic candidates -> Reciprocal Rank Fusion (RRF).
        If semantic search is disabled or unavailable, falls back safely to lexical retrieval.
        Strictly isolates chunks by user_id.
        """
        if not user_id or not query:
            return []

        # 1. Fetch lexical candidates
        lexical_candidates = self._retrieve_lexical_candidates(
            user_id=user_id, query=query, max_candidates=max_candidates
        )

        # 2. Check if semantic retrieval is active
        semantic_enabled = _get_config("AI_ENGINE_SEMANTIC_ENABLED", False)
        provider = self.get_embedding_provider() if semantic_enabled else None

        if not semantic_enabled or not provider:
            return lexical_candidates

        # 3. Generate query embedding and retrieve semantic candidates
        semantic_candidates: List[RetrievalResult] = []
        try:
            query_vec = provider.embed_text(query)
            top_k_sem = _get_config("AI_ENGINE_SEMANTIC_TOP_K", 25)
            semantic_candidates = self._retrieve_semantic_candidates(
                user_id=user_id, query_vec=query_vec, top_k=top_k_sem
            )
        except Exception as exc:
            logger.warning(
                "[DocumentService] Semantic query embedding failed for user=%s: %s (falling back to lexical only)",
                user_id, exc,
            )
            return lexical_candidates

        # 4. Fuse lexical and semantic candidates using Reciprocal Rank Fusion
        if semantic_candidates and lexical_candidates:
            fused = reciprocal_rank_fusion(
                lexical_candidates=lexical_candidates,
                semantic_candidates=semantic_candidates,
                limit=max_candidates,
            )
            return fused

        # If only one set produced results, return it
        return semantic_candidates if semantic_candidates else lexical_candidates
