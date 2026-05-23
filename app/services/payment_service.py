import razorpay
import hmac
import hashlib
from datetime import datetime, timedelta, timezone
from flask import current_app
from app.database.db import db
from app.models.payment import Payment
from app.models.user import User


PLANS = {
    "monthly": {
        "amount": 19900,       # ₹199 in paise
        "label": "Pro Monthly",
        "duration_days": 30,
    },
    "yearly": {
        "amount": 199900,      # ₹1999 in paise
        "label": "Pro Yearly",
        "duration_days": 365,
    },
}


def get_razorpay_client():
    return razorpay.Client(
        auth=(
            current_app.config["RAZORPAY_KEY_ID"],
            current_app.config["RAZORPAY_KEY_SECRET"],
        )
    )


def create_order(user_id: int, plan: str) -> dict:
    """Create a Razorpay order and persist it."""
    if plan not in PLANS:
        raise ValueError(f"Invalid plan: {plan}")

    client = get_razorpay_client()
    plan_info = PLANS[plan]

    rz_order = client.order.create(
        {
            "amount": plan_info["amount"],
            "currency": "INR",
            "payment_capture": 1,
            "notes": {
                "user_id": str(user_id),
                "plan": plan,
            },
        }
    )

    payment = Payment(
        user_id=user_id,
        razorpay_order_id=rz_order["id"],
        amount=plan_info["amount"],
        plan=plan,
        status="created",
    )
    db.session.add(payment)
    db.session.commit()

    return {
        "order_id": rz_order["id"],
        "amount": plan_info["amount"],
        "currency": "INR",
        "plan": plan,
        "key_id": current_app.config["RAZORPAY_KEY_ID"],
        # Tell Razorpay checkout to surface UPI options prominently
        "prefill_method": "upi",
    }


def verify_and_activate(
    razorpay_order_id: str,
    razorpay_payment_id: str,
    razorpay_signature: str,
) -> dict:
    """Verify HMAC signature, mark payment paid, upgrade user."""

    # 1. Signature check
    secret = current_app.config["RAZORPAY_KEY_SECRET"].encode()
    payload = f"{razorpay_order_id}|{razorpay_payment_id}".encode()
    expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(expected, razorpay_signature):
        raise ValueError("Invalid payment signature")

    # 2. Fetch payment record
    payment = Payment.query.filter_by(razorpay_order_id=razorpay_order_id).first()
    if not payment:
        raise ValueError("Order not found")

    # 3. Fetch extra details from Razorpay (method, UPI txn id)
    client = get_razorpay_client()
    rz_payment = client.payment.fetch(razorpay_payment_id)
    method = rz_payment.get("method", "unknown")
    upi_txn = rz_payment.get("vpa") or rz_payment.get("upi", {}).get("payer_account_type")

    # 4. Update payment record
    payment.razorpay_payment_id = razorpay_payment_id
    payment.razorpay_signature = razorpay_signature
    payment.status = "paid"
    payment.payment_method = method
    payment.upi_transaction_id = upi_txn if method == "upi" else None
    payment.paid_at = datetime.now(timezone.utc)

    # 5. Upgrade user
    user = User.query.get(payment.user_id)
    duration = PLANS[payment.plan]["duration_days"]

    # Extend if already premium (don't overwrite remaining days)
    now = datetime.now(timezone.utc)
    current_expiry = user.premium_expiry
    if current_expiry and current_expiry.tzinfo is None:
        current_expiry = current_expiry.replace(tzinfo=timezone.utc)
    base = max(current_expiry or now, now)
    user.is_premium = True
    user.premium_plan = payment.plan
    user.premium_expiry = base + timedelta(days=duration)

    db.session.commit()

    return {
        "status": "success",
        "plan": payment.plan,
        "expiry": user.premium_expiry.isoformat(),
        "method": method,
    }


def handle_webhook(payload: dict, webhook_signature: str, raw_body: bytes) -> None:
    """Process Razorpay webhooks (payment.failed, refund, etc.)."""
    secret = current_app.config.get("RAZORPAY_WEBHOOK_SECRET", "")
    if secret:
        expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, webhook_signature):
            raise ValueError("Invalid webhook signature")

    event = payload.get("event")

    if event == "payment.failed":
        order_id = payload["payload"]["payment"]["entity"]["order_id"]
        payment = Payment.query.filter_by(razorpay_order_id=order_id).first()
        if payment:
            payment.status = "failed"
            db.session.commit()

    elif event == "refund.created":
        payment_id = payload["payload"]["refund"]["entity"]["payment_id"]
        payment = Payment.query.filter_by(razorpay_payment_id=payment_id).first()
        if payment:
            payment.status = "refunded"
            user = User.query.get(payment.user_id)
            if user:
                user.is_premium = False
                user.premium_plan = None
                user.premium_expiry = None
            db.session.commit()
