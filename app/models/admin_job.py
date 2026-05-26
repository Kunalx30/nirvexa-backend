import uuid
from datetime import datetime, timezone
from app.extensions import db


class AdminJob(db.Model):
    """
    NirVexa Premium Job — Admin-posted job listings
    displayed exclusively to premium users on the Jobs page.
    """

    __tablename__ = "admin_jobs"

    id = db.Column(
        db.String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    # ── Job Details ──────────────────────────────────────────────
    title          = db.Column(db.String(200), nullable=False)
    company        = db.Column(db.String(200), nullable=False)
    company_logo   = db.Column(db.Text, nullable=True)   # URL or initial letter fallback
    location       = db.Column(db.String(200), nullable=True)
    job_type       = db.Column(db.String(50),  nullable=True, default="full-time")
    experience     = db.Column(db.String(100), nullable=True)   # e.g. "2–5 years"
    salary         = db.Column(db.String(100), nullable=True)   # e.g. "₹15–25 LPA"
    description    = db.Column(db.Text, nullable=True)
    requirements   = db.Column(db.Text, nullable=True)
    skills         = db.Column(db.ARRAY(db.String), nullable=True, default=list)

    # ── Apply ────────────────────────────────────────────────────
    apply_url      = db.Column(db.Text, nullable=True)
    apply_email    = db.Column(db.String(255), nullable=True)

    # ── Meta ─────────────────────────────────────────────────────
    is_active      = db.Column(db.Boolean, default=True, nullable=False)
    is_featured    = db.Column(db.Boolean, default=False, nullable=False)
    posted_by      = db.Column(db.String(100), nullable=True, default="NirVexa Team")
    category       = db.Column(db.String(100), nullable=True)   # e.g. "Engineering", "Design"

    # ── Timestamps ───────────────────────────────────────────────
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    def to_dict(self):
        return {
            "id":           self.id,
            "title":        self.title,
            "company":      self.company,
            "company_logo": self.company_logo,
            "location":     self.location,
            "job_type":     self.job_type,
            "experience":   self.experience,
            "salary":       self.salary,
            "description":  self.description,
            "requirements": self.requirements,
            "skills":       self.skills or [],
            "apply_url":    self.apply_url,
            "apply_email":  self.apply_email,
            "is_active":    self.is_active,
            "is_featured":  self.is_featured,
            "posted_by":    self.posted_by,
            "category":     self.category,
            "created_at":   self.created_at.isoformat(),
            "updated_at":   self.updated_at.isoformat(),
        }

    def __repr__(self):
        return f"<AdminJob {self.title} @ {self.company}>"
