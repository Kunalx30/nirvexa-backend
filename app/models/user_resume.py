"""
app/models/user_resume.py
Stores each AI-generated resume per user.

PLACEMENT: app/models/user_resume.py
IMPORT IN:  app/__init__.py  →  from app.models.user_resume import UserResume  # noqa
            app/models/__init__.py  →  from app.models.user_resume import UserResume
"""
import uuid
from datetime import datetime, timezone
from app.extensions import db


class UserResume(db.Model):
    __tablename__ = "user_resumes"

    id              = db.Column(db.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id         = db.Column(db.String, db.ForeignKey("users.id"), nullable=False)
    template_id = db.Column(db.UUID(as_uuid=True), db.ForeignKey("resume_templates.id", ondelete="SET NULL"), nullable=True)
    # All form fields the user submitted — stored as JSONB
    resume_data     = db.Column(db.JSON, nullable=False)

    # Groq AI output JSON — cached so user can re-download without re-calling AI
    ai_content      = db.Column(db.JSON)

    # Current LaTeX source — editable via live editor
    latex_draft     = db.Column(db.Text)

    # Cloudflare R2 URL — empty string if R2 not configured (local dev)
    pdf_url         = db.Column(db.String(500), default="")

    # Job description used (if JD-aware mode was used)
    jd_text         = db.Column(db.Text)

    # ATS scores (populated if JD was provided)
    ats_score       = db.Column(db.Integer)
    jd_match_score  = db.Column(db.Integer)

    # draft / compiled / downloaded
    status          = db.Column(db.String(20), default="draft")

    created_at      = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at      = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def to_dict(
        self,
        include_ai_content: bool = False,
        include_latex: bool = False,
    ) -> dict:
        d = {
            "id":             str(self.id),
            "user_id":        str(self.user_id),
            "template_id":    str(self.template_id) if self.template_id else None,
            "pdf_url":        self.pdf_url,
            "ats_score":      self.ats_score,
            "jd_match_score": self.jd_match_score,
            "status":         self.status,
            "created_at":     self.created_at.isoformat() if self.created_at else None,
            "updated_at":     self.updated_at.isoformat() if self.updated_at else None,
            # Include form data summary (not the full blob)
            "full_name":      (self.resume_data or {}).get("full_name", ""),
            "target_role":    (self.resume_data or {}).get("target_role", ""),
        }
        if include_ai_content:
            d["ai_content"] = self.ai_content
            d["resume_data"] = self.resume_data
        if include_latex:
            d["latex_draft"] = self.latex_draft
        return d