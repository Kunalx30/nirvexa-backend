import uuid
from datetime import datetime, timezone
from app.extensions import db


class User(db.Model):
    """
    NirVexa User Model
    Handles both email/password and Google OAuth users.
    """

    __tablename__ = "users"

    # --- Primary Key ---
    id = db.Column(
        db.String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    # --- Basic Info ---
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)

    # --- Auth Fields ---
    # NULL for Google users
    password_hash = db.Column(db.String(255), nullable=True)
    # NULL for email users
    google_id = db.Column(db.String(255), unique=True, nullable=True, index=True)

    # --- Profile ---
    avatar_url = db.Column(db.Text, nullable=True)
    skills = db.Column(db.ARRAY(db.String), nullable=True, default=list)
    preferred_location = db.Column(db.String(100), nullable=True)
    job_type = db.Column(
        db.String(50),
        nullable=True,
        default="full-time"
    )
    experience_level = db.Column(
        db.String(50),
        nullable=True,
        default="fresher"
    )

    # --- Auth Tokens ---
    refresh_token = db.Column(db.Text, nullable=True)

    # --- Status ---
    is_verified = db.Column(db.Boolean, default=False, nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)

    # --- Timestamps ---
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
    last_login = db.Column(db.DateTime(timezone=True), nullable=True)

    # ----------------------------------------
    # Helper Methods
    # ----------------------------------------

    def update_last_login(self):
        """Call this every time a user successfully logs in."""
        self.last_login = datetime.now(timezone.utc)

    def to_dict(self):
        """
        Safe public representation of the user.
        Never includes password_hash or refresh_token.
        """
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "avatar_url": self.avatar_url,
            "skills": self.skills or [],
            "preferred_location": self.preferred_location,
            "job_type": self.job_type,
            "experience_level": self.experience_level,
            "is_verified": self.is_verified,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }

    def __repr__(self):
        return f"<User {self.email}>"