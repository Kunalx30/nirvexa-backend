import uuid
from datetime import datetime, timezone
from app.database.db import db


class InterviewSession(db.Model):
    """One record per complete voice interview session."""

    __tablename__ = "interview_sessions"

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
    role = db.Column(db.String(100), nullable=False)        # e.g. "Data Analyst"
    mode = db.Column(db.String(50), nullable=False)         # hr / technical / stress / mock
    difficulty = db.Column(db.String(20), nullable=True)    # easy / medium / hard

    # Scores
    total_score = db.Column(db.Float, nullable=True)
    content_score = db.Column(db.Float, nullable=True)
    confidence_score = db.Column(db.Float, nullable=True)
    grammar_score = db.Column(db.Float, nullable=True)
    keyword_score = db.Column(db.Float, nullable=True)

    # Metrics
    filler_word_count = db.Column(db.Integer, nullable=True)
    avg_wpm = db.Column(db.Float, nullable=True)
    keywords_hit = db.Column(db.Integer, nullable=True)
    keywords_total = db.Column(db.Integer, nullable=True)
    total_questions = db.Column(db.Integer, nullable=True)

    # Status
    status = db.Column(
        db.String(20),
        nullable=False,
        default="in_progress"
    )  # in_progress / completed

    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    completed_at = db.Column(db.DateTime(timezone=True), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "role": self.role,
            "mode": self.mode,
            "difficulty": self.difficulty,
            "total_score": self.total_score,
            "content_score": self.content_score,
            "confidence_score": self.confidence_score,
            "grammar_score": self.grammar_score,
            "keyword_score": self.keyword_score,
            "filler_word_count": self.filler_word_count,
            "avg_wpm": self.avg_wpm,
            "keywords_hit": self.keywords_hit,
            "keywords_total": self.keywords_total,
            "total_questions": self.total_questions,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    def __repr__(self):
        return f"<InterviewSession {self.role} — {self.mode} — {self.status}>"