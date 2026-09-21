"""
app/models/ai_document.py
SQLAlchemy models for Phase 4 user-isolated document knowledge persistence.
Stores user uploaded documents and their extracted evidence chunks.
"""
import uuid
from datetime import datetime, timezone
from app.extensions import db


class AIDocument(db.Model):
    """
    Metadata for user-uploaded private knowledge documents.
    Enforces multi-tenant isolation via user_id.
    """
    __tablename__ = "ai_documents"

    id = db.Column(
        db.String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id = db.Column(
        db.String(36),
        db.ForeignKey("users.id"),
        nullable=False,
        index=True,
    )
    filename = db.Column(db.String(255), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(20), nullable=False)  # pdf, txt, md
    file_size = db.Column(db.Integer, nullable=False)     # in bytes
    chunk_count = db.Column(db.Integer, default=0)

    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    chunks = db.relationship(
        "AIDocumentChunk",
        backref="document",
        cascade="all, delete-orphan",
        lazy="select",
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "filename": self.filename,
            "title": self.title,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "chunk_count": self.chunk_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AIDocumentChunk(db.Model):
    """
    Granular text chunks extracted from an AIDocument.
    Denormalizes user_id for high-performance scoped filtering and multi-tenant security.
    """
    __tablename__ = "ai_document_chunks"

    id = db.Column(
        db.String(64),
        primary_key=True,
        default=lambda: f"chk_{uuid.uuid4().hex[:16]}",
    )
    document_id = db.Column(
        db.String(36),
        db.ForeignKey("ai_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = db.Column(
        db.String(36),
        db.ForeignKey("users.id"),
        nullable=False,
        index=True,
    )
    chunk_index = db.Column(db.Integer, nullable=False)
    text = db.Column(db.Text, nullable=False)
    char_count = db.Column(db.Integer, nullable=False)
    metadata_json = db.Column(db.JSON, nullable=True)

    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "user_id": self.user_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "char_count": self.char_count,
            "metadata": self.metadata_json or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
