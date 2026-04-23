from app.extensions import db
import uuid
from datetime import datetime

class SavedJob(db.Model):
    __tablename__ = 'saved_jobs'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    job_id = db.Column(db.String(36), db.ForeignKey('jobs.id'), nullable=False)
    status = db.Column(db.String(50), default='Saved')  # Saved/Applied/Interview/Offer/Rejected
    notes = db.Column(db.Text)
    saved_at = db.Column(db.DateTime, default=datetime.utcnow)