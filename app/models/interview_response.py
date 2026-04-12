import uuid
from datetime import datetime, timezone
from app.database.db import db


class InterviewResponse(db.Model):
    """One record per question answered inside a session."""

    __tablename__ = "interview_responses"

    id = db.Column(
        db.String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    session_id = db.Column(
        db.String(36),
        db.ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    question_number = db.Column(db.Integer, nullable=False)
    question_text = db.Column(db.Text, nullable=False)

    # User's spoken answer
    transcript = db.Column(db.Text, nullable=True)
    duration_seconds = db.Column(db.Integer, nullable=True)
    word_count = db.Column(db.Integer, nullable=True)
    filler_words_detected = db.Column(db.ARRAY(db.String), nullable=True)

    # AI evaluation
    score = db.Column(db.Float, nullable=True)
    content_score = db.Column(db.Float, nullable=True)
    grammar_score = db.Column(db.Float, nullable=True)
    confidence_score = db.Column(db.Float, nullable=True)
    feedback = db.Column(db.Text, nullable=True)
    suggested_answer = db.Column(db.Text, nullable=True)
    keywords_hit = db.Column(db.ARRAY(db.String), nullable=True)
    keywords_missing = db.Column(db.ARRAY(db.String), nullable=True)

    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "question_number": self.question_number,
            "question_text": self.question_text,
            "transcript": self.transcript,
            "duration_seconds": self.duration_seconds,
            "word_count": self.word_count,
            "filler_words_detected": self.filler_words_detected or [],
            "score": self.score,
            "content_score": self.content_score,
            "grammar_score": self.grammar_score,
            "confidence_score": self.confidence_score,
            "feedback": self.feedback,
            "suggested_answer": self.suggested_answer,
            "keywords_hit": self.keywords_hit or [],
            "keywords_missing": self.keywords_missing or [],
            "created_at": self.created_at.isoformat(),
        }

    def __repr__(self):
        return f"<InterviewResponse Q{self.question_number} — session {self.session_id}>"