import jwt
from flask import Blueprint, jsonify, request, g

from app.middleware.rate_limiter import get_current_user, get_usage_summary
from app.utils.helpers import decode_token

usage_bp = Blueprint("usage", __name__, url_prefix="/api/user")


def _default_usage_payload():
    return {
        "is_premium": False,
        "premium_expiry": None,
        "usage": {
            "chat": {"limit": 7, "used": 0, "remaining": 7},
            "resume_analysis": {"limit": 1, "used": 0, "remaining": 1},
            "resume_build": {"limit": 0, "used": 0, "remaining": 0},
            "interview": {"limit": 3, "used": 0, "remaining": 3},
            "career_roadmap": {"limit": 3, "used": 0, "remaining": 3},
            "roadmap_search": {"limit": 3, "used": 0, "remaining": 3},
            "job_apply": {"limit": 10, "used": 0, "remaining": 10},
            "skill_match": {"limit": 3, "used": 0, "remaining": 3},
            "company_research": {"limit": 2, "used": 0, "remaining": 2},
            "salary_insights": {"limit": 2, "used": 0, "remaining": 2},
            "news": {"limit": 20, "used": 0, "remaining": 20},
        },
    }


def _load_optional_user():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None

    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        return None

    try:
        payload = decode_token(token)
        if payload.get("type") != "access":
            return None
        g.user_id = payload["sub"]
        return get_current_user()
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError, KeyError):
        return None


@usage_bp.route("/usage", methods=["GET"])
def usage_summary():
    current_user = _load_optional_user()

    if current_user is None:
        return jsonify(_default_usage_payload()), 200

    try:
        summary = get_usage_summary(current_user)

        is_premium = getattr(current_user, "is_premium", False) or False
        premium_expiry = getattr(current_user, "premium_expiry", None)

        return jsonify({
            "is_premium": is_premium,
            "premium_expiry": premium_expiry.isoformat() if premium_expiry else None,
            "usage": summary,
        }), 200

    except Exception as e:
        # Fallback: safe defaults so frontend never breaks.
        payload = _default_usage_payload()
        payload["error_detail"] = str(e)
        return jsonify(payload), 200
