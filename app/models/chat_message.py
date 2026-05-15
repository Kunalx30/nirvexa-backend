import uuid
from datetime import datetime, timezone
from app.extensions import db


class ChatMessage(db.Model):
    """Stores conversation history per user for AI memory."""

    __tablename__ = "chat_history"

    id = db.Column(
        db.String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    user_id = db.Column(
        db.String(36),
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    session_id = db.Column(
        db.String(36),
        db.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )
    role = db.Column(
        db.String(20),
        nullable=False
    )  # "user" or "assistant"
    content = db.Column(db.Text, nullable=False)
    model_used = db.Column(db.String(50), nullable=True)  # which LLM responded
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    session = db.relationship("ChatSession", back_populates="messages")

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "model_used": self.model_used,
            "created_at": self.created_at.isoformat(),
        }

    def __repr__(self):
        return f"<ChatMessage {self.role} — {self.user_id}>"
