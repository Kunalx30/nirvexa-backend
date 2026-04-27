import uuid
from datetime import datetime, timezone
from app.database.db import db


class NewsCache(db.Model):
    __tablename__ = 'news_cache'

    id = db.Column(db.String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Core article fields
    title       = db.Column(db.String(500), nullable=False)
    source      = db.Column(db.String(100), nullable=False)
    url         = db.Column(db.String(1000), nullable=False, unique=True)  # dedup key
    published_at = db.Column(db.DateTime, nullable=True)
    category    = db.Column(db.String(50), nullable=False, default='general')

    # AI generated 2-line summary by Mistral 7B
    ai_summary  = db.Column(db.Text, nullable=True)

    # When we cached it
    cached_at   = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            'id':           self.id,
            'title':        self.title,
            'source':       self.source,
            'url':          self.url,
            'published_at': self.published_at.isoformat() if self.published_at else None,
            'category':     self.category,
            'ai_summary':   self.ai_summary,
            'cached_at':    self.cached_at.isoformat() if self.cached_at else None,
        }