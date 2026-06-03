"""
app/models/interview_response.py

FIXES APPLIED:
  - FIX #5:  Replaced db.ARRAY(db.String) with db.JSON for filler_words_detected,
             keywords_hit, and keywords_missing.

             db.ARRAY is PostgreSQL-only.  db.JSON works on PostgreSQL, MySQL,
             SQLite (3.38+), and any other dialect SQLAlchemy supports.

             The stored values are Python lists — JSON serialises and deserialises
             them transparently so no other code needs to change.  Your existing
             .to_dict() already does `or []` on these fields, so it continues to
             work as-is.

  MIGRATION NOTE:
    If you have existing rows with ARRAY columns you must migrate the column type.
    Run this Alembic migration after deploying:

        from alembic import op
        import sqlalchemy as sa

        def upgrade():
            op.alter_column('interview_responses', 'filler_words_detected',
                            existing_type=sa.ARRAY(sa.String()),
                            type_=sa.JSON(),
                            postgresql_using='to_json(filler_words_detected)')
            op.alter_column('interview_responses', 'keywords_hit',
                            existing_type=sa.ARRAY(sa.String()),
                            type_=sa.JSON(),
                            postgresql_using='to_json(keywords_hit)')
            op.alter_column('interview_responses', 'keywords_missing',
                            existing_type=sa.ARRAY(sa.String()),
                            type_=sa.JSON(),
                            postgresql_using='to_json(keywords_missing)')

        def downgrade():
            # Reverse if needed — omitted for brevity.
            pass

    If this is a fresh database (no production data yet) you can skip the
    migration and just let db.create_all() / Alembic autogenerate create the
    table fresh with JSON columns.
"""
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

    # FIX #5: Changed from db.ARRAY(db.String) to db.JSON.
    # db.ARRAY is a PostgreSQL extension — it silently breaks on SQLite (CI)
    # and any non-Postgres deployment.  db.JSON stores the list as a JSON
    # string, which every supported dialect handles natively.
    # Read/write behaviour is identical: assign a Python list, get a list back.
    filler_words_detected = db.Column(db.JSON, nullable=True)

    # AI evaluation
    score = db.Column(db.Float, nullable=True)
    content_score = db.Column(db.Float, nullable=True)
    grammar_score = db.Column(db.Float, nullable=True)
    confidence_score = db.Column(db.Float, nullable=True)
    feedback = db.Column(db.Text, nullable=True)
    suggested_answer = db.Column(db.Text, nullable=True)

    # FIX #5: Same change — JSON instead of ARRAY(String)
    keywords_hit = db.Column(db.JSON, nullable=True)
    keywords_missing = db.Column(db.JSON, nullable=True)

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