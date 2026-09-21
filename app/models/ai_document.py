"""
app/models/ai_document.py
SQLAlchemy models for Phase 4 user-isolated document knowledge persistence.
Stores user uploaded documents and their extracted evidence chunks.
"""
import json
import uuid
from datetime import datetime, timezone
from sqlalchemy.types import TypeDecorator, UserDefinedType
from sqlalchemy import Text
from app.extensions import db


class VectorType(TypeDecorator):
    """
    SQLAlchemy TypeDecorator for vector embeddings.
    - PostgreSQL: Compiles to native `vector(dim)` supported by pgvector extension.
    - SQLite / Other: Compiles to Text/JSON, preserving in-memory testing compatibility.
    """
    impl = Text
    cache_ok = True

    def __init__(self, dimension: int = 768, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dimension = dimension

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            class _PGVector(UserDefinedType):
                def __init__(self, dim: int):
                    self.dim = dim

                def get_col_spec(self, **kw):
                    return f"vector({self.dim})"

            return dialect.type_descriptor(_PGVector(self.dimension))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if dialect.name == "postgresql":
                joined = ",".join(str(float(x)) for x in value)
                return f"[{joined}]"
            return json.dumps([float(x) for x in value])
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            val = value.strip()
            if val.startswith("[") and val.endswith("]"):
                inner = val[1:-1].strip()
                if not inner:
                    return []
                try:
                    return [float(x.strip()) for x in inner.split(",")]
                except (ValueError, TypeError):
                    pass
            try:
                return json.loads(val)
            except Exception:
                return None
        return value


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
    Includes nullable vector embedding for Phase 5 semantic retrieval.
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
    embedding = db.Column(VectorType(dimension=768), nullable=True)

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
            "has_embedding": self.embedding is not None,
            "metadata": self.metadata_json or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
