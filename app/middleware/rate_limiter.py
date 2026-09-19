import os
"""IP throttles plus per-user feature quotas."""

from datetime import date, datetime, timezone
from functools import wraps

from flask import g, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sqlalchemy import text

from app.database.db import db
from app.models.user import User


limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=os.getenv("REDIS_URL") or os.getenv("RATELIMIT_STORAGE_URL", "memory://"),
)


FREE_LIMITS = {
    "chat": 7,
    "resume_analysis": 1,
    "resume_build": 0,
    "interview": 3,
    "career_roadmap": 3,
    "roadmap_search": 3,
    "job_apply": 10,
    "skill_match": 3,
    "company_research": 2,
    "salary_insights": 2,
    "news": 20,
}

FEATURE_LABELS = {
    "chat": "AI Chat messages",
    "resume_analysis": "Resume ATS analyses",
    "resume_build": "Resume Builder exports",
    "interview": "Interview Sessions",
    "career_roadmap": "Career Roadmaps",
    "roadmap_search": "Roadmap searches",
    "job_apply": "Job Applications",
    "skill_match": "Skill Match",
    "company_research": "Company Research searches",
    "salary_insights": "Salary Insight searches",
    "news": "News Feed",
}


def _ensure_table():
    if db.engine.dialect.name == "postgresql":
        ddl = """
            CREATE TABLE IF NOT EXISTS feature_usage (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(36) NOT NULL,
                feature VARCHAR(50) NOT NULL,
                used_date DATE NOT NULL,
                count INTEGER DEFAULT 0,
                UNIQUE(user_id, feature, used_date)
            )
        """
    else:
        ddl = """
            CREATE TABLE IF NOT EXISTS feature_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id VARCHAR(36) NOT NULL,
                feature VARCHAR(50) NOT NULL,
                used_date DATE NOT NULL,
                count INTEGER DEFAULT 0,
                UNIQUE(user_id, feature, used_date)
            )
        """
    db.session.execute(text(ddl))
    db.session.commit()


def _get_usage(user_id: str, feature: str) -> int:
    row = db.session.execute(
        text(
            "SELECT count FROM feature_usage "
            "WHERE user_id=:u AND feature=:f AND used_date=:d"
        ),
        {"u": str(user_id), "f": feature, "d": date.today().isoformat()},
    ).fetchone()
    return row[0] if row else 0


def _increment_usage(user_id: str, feature: str):
    db.session.execute(
        text(
            """
            INSERT INTO feature_usage (user_id, feature, used_date, count)
            VALUES (:u, :f, :d, 1)
            ON CONFLICT(user_id, feature, used_date)
            DO UPDATE SET count = feature_usage.count + 1
            """
        ),
        {"u": str(user_id), "f": feature, "d": date.today().isoformat()},
    )
    db.session.commit()


def _is_premium_active(user) -> bool:
    if not user or not getattr(user, "is_premium", False):
        return False

    expiry = getattr(user, "premium_expiry", None)
    now = datetime.now(timezone.utc)
    if expiry and expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)

    if expiry and expiry < now:
        user.is_premium = False
        user.premium_plan = None
        db.session.commit()
        return False

    return True


def check_premium_status(user_id, feature: str = "premium") -> bool:
    """Check if the user has an active premium subscription."""
    if not user_id:
        return False
    user = User.query.get(str(user_id))
    return _is_premium_active(user)



def get_current_user():
    user = getattr(g, "current_user", None)
    if user:
        return user

    user_id = getattr(g, "user_id", None)
    if not user_id:
        return None

    user = User.query.get(str(user_id))
    g.current_user = user
    return user


def rate_limit(feature: str):
    """Enforce a per-feature daily free limit. Apply after @token_required."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            current_user = kwargs.get("current_user") or get_current_user()
            if current_user is None:
                return jsonify({"error": "Unauthorized"}), 401

            if _is_premium_active(current_user):
                _increment_usage(current_user.id, feature)
                return fn(*args, **kwargs)

            limit = FREE_LIMITS.get(feature, 10)
            used = _get_usage(current_user.id, feature)

            if used >= limit:
                label = FEATURE_LABELS.get(feature, feature)
                return jsonify({
                    "error": "rate_limit_exceeded",
                    "message": f"You have reached your daily free limit of {limit} {label}.",
                    "feature": feature,
                    "limit": limit,
                    "used": used,
                    "upgrade_url": "/pricing",
                }), 429

            _increment_usage(current_user.id, feature)
            return fn(*args, **kwargs)

        return wrapper
    return decorator


def premium_required(feature: str = "premium"):
    """Block free users from premium-only actions."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            current_user = kwargs.get("current_user") or get_current_user()
            if current_user is None:
                return jsonify({"error": "Unauthorized"}), 401
            if not _is_premium_active(current_user):
                return jsonify({
                    "error": "premium_required",
                    "message": "This feature is available on Nyrvexa Pro.",
                    "feature": feature,
                    "upgrade_url": "/pricing",
                }), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def get_usage_summary(user) -> dict:
    """Return remaining quota for all features."""
    if _is_premium_active(user):
        return {
            feature: {
                "limit": "unlimited",
                "used": _get_usage(user.id, feature),
                "remaining": "unlimited",
            }
            for feature in FREE_LIMITS
        }

    summary = {}
    for feature, limit in FREE_LIMITS.items():
        used = _get_usage(user.id, feature)
        summary[feature] = {
            "limit": limit,
            "used": used,
            "remaining": max(0, limit - used),
        }
    return summary
