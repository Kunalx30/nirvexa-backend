import json

import requests
from flask import Blueprint, g, jsonify, redirect, request, url_for

from app.database.db import db
from app.middleware.auth_middleware import token_required
from app.models.interview_session import InterviewSession
from app.models.resume_analysis import ResumeAnalysis
from app.models.user import User
from app.models.user_resume import UserResume


profile_bp = Blueprint("profile", __name__, url_prefix="/api/profile")


def _clean_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _clean_url(value):
    value = (value or "").strip()
    if not value:
        return None
    if value.startswith(("http://", "https://")):
        return value[:255]
    return f"https://{value}"[:255]


def _score(session, *names, default=None):
    for name in names:
        value = getattr(session, name, None)
        if value is not None:
            return round(float(value), 1)
    return default


def _latest_interview_scores(user_id):
    session = (
        InterviewSession.query
        .filter_by(user_id=user_id)
        .order_by(InterviewSession.completed_at.desc().nullslast(), InterviewSession.created_at.desc())
        .first()
    )
    if not session:
        return None

    return {
        "content": _score(session, "score_content", "content_score", "total_score", default=0),
        "clarity": _score(session, "score_clarity", "grammar_score", "total_score", default=0),
        "depth": _score(session, "score_depth", "keyword_score", "total_score", default=0),
        "relevance": _score(session, "score_relevance", "total_score", default=0),
        "confidence": _score(session, "score_confidence", "confidence_score", default=0),
        "structure": _score(session, "score_structure", "total_score", default=0),
        "session_date": (session.completed_at or session.created_at).isoformat() if (session.completed_at or session.created_at) else None,
    }


def _parse_raw_response(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except Exception:
                return {}
    return {}


def _latest_skill_match(user):
    analysis = (
        ResumeAnalysis.query
        .filter_by(user_id=user.id)
        .order_by(ResumeAnalysis.created_at.desc())
        .first()
    )
    if not analysis:
        return None

    raw = _parse_raw_response(analysis.raw_response)
    matched = raw.get("matched_keywords") or analysis.skills_found or []
    missing = raw.get("missing_keywords") or analysis.missing_keywords or []
    role = (user.target_roles or [None])[0] or raw.get("role_match") or "Target role"

    return {
        "role": role,
        "match_pct": analysis.score or raw.get("jd_match_score") or raw.get("overall_score") or 0,
        "matched": matched,
        "missing": missing,
    }


def _latest_public_resume(user):
    return (
        UserResume.query
        .filter_by(user_id=user.id)
        .filter(UserResume.pdf_url.isnot(None))
        .filter(UserResume.pdf_url != "")
        .order_by(UserResume.updated_at.desc(), UserResume.created_at.desc())
        .first()
    )


def _profile_strength(user):
    checks = [
        bool(user.name),
        bool(user.bio),
        bool(user.skills),
        bool(user.target_roles),
        bool(user.location_label or user.preferred_location),
        bool(user.linkedin_url or user.github_url or user.portfolio_url),
    ]
    return round((sum(checks) / len(checks)) * 100)


def _public_payload(user):
    resume = _latest_public_resume(user) if user.show_resume_download else None
    payload = {
        "username": user.username,
        "full_name": user.name,
        "level": user.experience_level,
        "bio": user.bio,
        "location": user.location_label or user.preferred_location,
        "linkedin_url": user.linkedin_url,
        "github_url": user.github_url,
        "portfolio_url": user.portfolio_url,
        "skills": user.skills or [],
        "target_roles": user.target_roles or [],
        "theme_gradient": user.theme_gradient or user.avatar_url,
        "profile_strength": _profile_strength(user),
        "show_interview_scores": user.show_interview_scores,
        "show_skill_match": user.show_skill_match,
        "show_resume_download": user.show_resume_download,
        "interview_scores": None,
        "skill_match": None,
        "resume_url": url_for("profile.download_public_resume", username=user.username, _external=True) if resume else None,
    }

    if user.show_interview_scores:
        payload["interview_scores"] = _latest_interview_scores(user.id)
    if user.show_skill_match:
        payload["skill_match"] = _latest_skill_match(user)

    return payload


@profile_bp.route("/public/<username>", methods=["GET"])
def get_public_profile(username):
    user = User.query.filter_by(username=(username or "").strip().lower()).first()
    if not user:
        return jsonify({"error": "Profile not found"}), 404
    if not user.public_profile_enabled:
        return jsonify({"error": "Profile is private"}), 403
    return jsonify(_public_payload(user)), 200


@profile_bp.route("/public/<username>/resume", methods=["GET"])
def download_public_resume(username):
    user = User.query.filter_by(username=(username or "").strip().lower()).first()
    if not user:
        return jsonify({"error": "Profile not found"}), 404
    if not user.public_profile_enabled or not user.show_resume_download:
        return jsonify({"error": "Resume download is not enabled"}), 403

    resume = _latest_public_resume(user)
    if not resume:
        return jsonify({"error": "Resume not found"}), 404

    try:
        head = requests.head(resume.pdf_url, allow_redirects=True, timeout=5)
        if head.status_code >= 400:
            return jsonify({"error": "Resume is unavailable"}), 404
    except requests.RequestException:
        pass

    return redirect(resume.pdf_url, code=302)


@profile_bp.route("/settings", methods=["PATCH"])
@token_required
def update_public_profile_settings():
    data = request.get_json(silent=True) or {}
    user = User.query.filter_by(id=g.user_id).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    if not user.username:
        user.username = User.generate_unique_username(user.name, current_user_id=user.id)

    for field in ("bio", "location_label"):
        if field in data:
            value = (data.get(field) or "").strip()
            setattr(user, field, value or None)

    for field in ("linkedin_url", "github_url", "portfolio_url"):
        if field in data:
            setattr(user, field, _clean_url(data.get(field)))

    for field in ("skills", "target_roles"):
        if field in data:
            setattr(user, field, _clean_list(data.get(field)))

    for field in (
        "public_profile_enabled",
        "show_interview_scores",
        "show_skill_match",
        "show_resume_download",
    ):
        if field in data:
            setattr(user, field, bool(data.get(field)))

    db.session.commit()
    return jsonify({"success": True, "user": user.to_dict()}), 200
