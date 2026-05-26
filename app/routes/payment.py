from flask import Blueprint, request, jsonify
from app.middleware.auth_middleware import token_required
from app.middleware.rate_limiter import get_current_user
from app.services.payment_service import create_order, verify_and_activate, handle_webhook, get_plans_catalog

payment_bp = Blueprint("payment", __name__, url_prefix="/api/payment")


@payment_bp.route("/plans", methods=["GET"])
def list_plans():
    """Public pricing catalog for the marketing and checkout UI."""
    return jsonify(get_plans_catalog()), 200


@payment_bp.route("/create-order", methods=["POST"])
@token_required
def create_payment_order():
    """
    POST /api/payment/create-order
    Body: { "plan": "monthly" | "yearly" }
    """
    data = request.get_json() or {}
    plan = data.get("plan", "monthly")
    current_user = get_current_user()

    try:
        order = create_order(user_id=current_user.id, plan=plan)
        return jsonify(order), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Could not create order", "detail": str(e)}), 500


@payment_bp.route("/verify", methods=["POST"])
@token_required
def verify_payment():
    """
    POST /api/payment/verify
    Body: {
        "razorpay_order_id": "...",
        "razorpay_payment_id": "...",
        "razorpay_signature": "..."
    }
    """
    data = request.get_json() or {}
    required = ["razorpay_order_id", "razorpay_payment_id", "razorpay_signature"]
    missing = [k for k in required if not data.get(k)]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        result = verify_and_activate(
            razorpay_order_id=data["razorpay_order_id"],
            razorpay_payment_id=data["razorpay_payment_id"],
            razorpay_signature=data["razorpay_signature"],
        )
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Verification failed", "detail": str(e)}), 500


@payment_bp.route("/webhook", methods=["POST"])
def razorpay_webhook():
    """
    Razorpay calls this for payment.failed, refund.created, etc.
    Set this URL in Razorpay Dashboard → Webhooks.
    """
    signature = request.headers.get("X-Razorpay-Signature", "")
    raw_body = request.get_data()
    payload = request.get_json(silent=True) or {}

    try:
        handle_webhook(payload, signature, raw_body)
        return jsonify({"status": "ok"}), 200
    except ValueError:
        return jsonify({"error": "Invalid webhook"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@payment_bp.route("/status", methods=["GET"])
@token_required
def premium_status():
    current_user = get_current_user()
    """GET /api/payment/status — returns current user's subscription state."""
    return jsonify({
        "is_premium": current_user.is_premium,
        "plan": current_user.premium_plan,
        "expiry": current_user.premium_expiry.isoformat() if current_user.premium_expiry else None,
    }), 200
