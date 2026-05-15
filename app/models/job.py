from app.extensions import db
import uuid
from datetime import datetime, timedelta

class Job(db.Model):
    __tablename__ = 'jobs'
    __table_args__ = (
        db.UniqueConstraint('title', 'company', 'location', name='uq_job_title_company_location'),
    )

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = db.Column(db.String(255), nullable=False)
    company = db.Column(db.String(255), nullable=False)
    location = db.Column(db.String(255))
    skills = db.Column(db.ARRAY(db.String))
    salary = db.Column(db.String(100))
    source = db.Column(db.String(100))
    job_type = db.Column(db.String(50))       # fulltime / internship / remote / freelance
    apply_url = db.Column(db.Text)
    description = db.Column(db.Text)
    posted_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, default=lambda: datetime.utcnow() + timedelta(days=30))
    is_active = db.Column(db.Boolean, default=True)
    is_fresher = db.Column(db.Boolean, default=False, index=True)  # ← ADD THIS
    ai_summary = db.Column(db.Text)