from app.database.db import db
from datetime import datetime


class Payment(db.Model):
    __tablename__ = "payments"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String, db.ForeignKey("users.id"), nullable=False)

    razorpay_order_id = db.Column(db.String(100), unique=True, nullable=False)
    razorpay_payment_id = db.Column(db.String(100), unique=True, nullable=True)
    razorpay_signature = db.Column(db.String(256), nullable=True)

    amount = db.Column(db.Integer, nullable=False)          # in paise (e.g. 49900 = ₹499)
    currency = db.Column(db.String(10), default="INR")
    plan = db.Column(db.String(50), nullable=False)         # "monthly" | "yearly"
    status = db.Column(db.String(30), default="created")   # created | paid | failed

    payment_method = db.Column(db.String(50), nullable=True)  # upi | card | netbanking | wallet
    upi_transaction_id = db.Column(db.String(100), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    paid_at = db.Column(db.DateTime, nullable=True)

    user = db.relationship("User", backref="payments")

    def to_dict(self):
        return {
            "id": self.id,
            "razorpay_order_id": self.razorpay_order_id,
            "razorpay_payment_id": self.razorpay_payment_id,
            "amount": self.amount,
            "currency": self.currency,
            "plan": self.plan,
            "status": self.status,
            "payment_method": self.payment_method,
            "created_at": self.created_at.isoformat(),
            "paid_at": self.paid_at.isoformat() if self.paid_at else None,
        }