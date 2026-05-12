"""
app/models/resume_template.py
Stores the 6 predefined LaTeX resume templates.

PLACEMENT: app/models/resume_template.py
IMPORT IN:  app/__init__.py  →  from app.models.resume_template import ResumeTemplate  # noqa
            app/models/__init__.py  →  from app.models.resume_template import ResumeTemplate
"""
import uuid
from datetime import datetime, timezone
from app.extensions import db


class ResumeTemplate(db.Model):
    __tablename__ = "resume_templates"

    id          = db.Column(db.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name        = db.Column(db.String(100), nullable=False)
    category    = db.Column(db.String(50))          # tech / minimal / classic / creative
    best_for    = db.Column(db.String(200))
    description = db.Column(db.Text)
    preview_url = db.Column(db.String(500), default="")
    latex_code  = db.Column(db.Text, nullable=False)
    # slug = the "id" string from metadata.json e.g. "template_01_modern_blue"
    # Used for block-routing in resume_builder_service.py — never changes
    slug        = db.Column(db.String(100), unique=True)
    is_active   = db.Column(db.Boolean, default=True)
    created_at  = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self, include_latex: bool = False) -> dict:
        d = {
            "id":          str(self.id),
            "name":        self.name,
            "category":    self.category,
            "best_for":    self.best_for,
            "description": self.description,
            "preview_url": self.preview_url,
            "slug":        self.slug,
            "is_active":   self.is_active,
            "created_at":  self.created_at.isoformat() if self.created_at else None,
        }
        if include_latex:
            d["latex_code"] = self.latex_code
        return d