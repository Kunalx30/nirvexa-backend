import uuid
from datetime import datetime, timezone
from app.extensions import db


class SupportTicket(db.Model):
    """
    NirVexa Support Ticket Model
    Stores user-submitted support queries for admin review.
    """

    __tablename__ = "support_tickets"

    # --- Primary Key ---
    id = db.Column(
        db.String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    # --- Ticket Reference ---
    ticket_ref = db.Column(db.String(20), unique=True, nullable=False)  # e.g. NVX-8910

    # --- User Info ---
    user_id = db.Column(
        db.String(36),
        db.ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )
    user_name = db.Column(db.String(100), nullable=True)
    user_email = db.Column(db.String(255), nullable=False)

    # --- Ticket Content ---
    subject = db.Column(db.String(100), nullable=False, default="general")
    message = db.Column(db.Text, nullable=False)

    # --- Status ---
    status = db.Column(
        db.String(30),
        nullable=False,
        default="open"
    )  # open | in_progress | resolved | closed

    # --- Admin Notes ---
    admin_notes = db.Column(db.Text, nullable=True)

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

    # Relationship back to User
    user = db.relationship("User", backref=db.backref("support_tickets", lazy="dynamic"))

    def to_dict(self):
        return {
            "id": self.id,
            "ticket_ref": self.ticket_ref,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "user_email": self.user_email,
            "subject": self.subject,
            "message": self.message,
            "status": self.status,
            "admin_notes": self.admin_notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def __repr__(self):
        return f"<SupportTicket {self.ticket_ref} [{self.status}]>"
