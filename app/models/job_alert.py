from app.extensions import db
import uuid
from datetime import datetime

class JobAlert(db.Model):
    __tablename__ = 'job_alerts'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    keywords = db.Column(db.ARRAY(db.String))
    location = db.Column(db.String(255))
    frequency = db.Column(db.String(50), default='daily')
    last_sent_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)