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
    is_active = db.Column(db.Boolean, default=True)
    last_sent_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def keywords_list(self) -> list:
        """Return keywords as a clean lowercase list."""
        return [k.strip().lower() for k in (self.keywords or []) if k.strip()]