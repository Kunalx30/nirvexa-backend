"""
app/models/resume_analysis.py
Stores resume analysis results linked to a user.
"""
import uuid
from datetime import datetime, timezone
from app.extensions import db

class ResumeAnalysis(db.Model):
    __tablename__ = "resume_analyses"

    id         = db.Column(db.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = db.Column(db.String, db.ForeignKey("users.id"), nullable=False)
    filename   = db.Column(db.String(255))
    score      = db.Column(db.Integer)
    ats_status = db.Column(db.String(50))
    skills_found      = db.Column(db.ARRAY(db.String))
    strengths         = db.Column(db.ARRAY(db.String))
    improvements      = db.Column(db.ARRAY(db.String))
    missing_keywords  = db.Column(db.ARRAY(db.String))
    raw_response      = db.Column(db.Text)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))