import os
import random
from datetime import datetime, timezone
from functools import wraps

from flask import Blueprint, jsonify, request, g, current_app

from app.extensions import db
from app.models.user import User
from app.models.support_ticket import SupportTicket
from app.models.admin_job import AdminJob
from app.middleware.team_admin_middleware import team_admin_required
from app.middleware.rate_limiter import limiter
from app.services.team_admin_auth import (
    team_admin_email,
    is_team_admin_configured,
    is_allowed_admin_email,
    verify_password,
    issue_otp,
    verify_otp,
    generate_team_admin_token,
    decode_team_admin_token,
)
import jwt
from app.services.email_service import send_team_admin_login_alert, send_team_admin_otp
from app.utils.helpers import success_response, error_response
from app.utils.debug_log import agent_log

admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")

# ─────────────────────────────────────────────
# Auth guard — reads ADMIN_SECRET_KEY from env
# ─────────────────────────────────────────────

def require_admin(f):
    """Team admin JWT (from /teamadmin) or X-Admin-Key header."""
    @wraps(f)
    def decorated(*args, **kwargs):
        secret = os.getenv("ADMIN_SECRET_KEY", "")
        provided = request.headers.get("X-Admin-Key", "")
        if secret and provided == secret:
            return f(*args, **kwargs)

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                payload = decode_team_admin_token(auth_header.split(" ", 1)[1])
                g.team_admin_email = payload["sub"]
                return f(*args, **kwargs)
            except jwt.ExpiredSignatureError:
                return error_response("Session expired. Please sign in again.", 401)
            except jwt.InvalidTokenError:
                return error_response("Invalid team admin session", 401)

        return error_response("Team admin authentication required", 401)
    return decorated


# ─────────────────────────────────────────────
# Support Ticket Submission  (public endpoint)
# ─────────────────────────────────────────────

@admin_bp.route("/tickets", methods=["POST"])
def submit_ticket():
    """Allow logged-in (or guest) users to submit support tickets."""
    data = request.get_json(silent=True) or {}

    user_email = data.get("user_email", "").strip()
    message    = data.get("message", "").strip()

    if not user_email or not message:
        return jsonify({"error": "user_email and message are required"}), 400

    ticket_ref = f"NVX-{random.randint(1000, 9999)}"
    # Ensure uniqueness
    while SupportTicket.query.filter_by(ticket_ref=ticket_ref).first():
        ticket_ref = f"NVX-{random.randint(1000, 9999)}"

    ticket = SupportTicket(
        ticket_ref=ticket_ref,
        user_id=data.get("user_id"),
        user_name=data.get("user_name", ""),
        user_email=user_email,
        subject=data.get("subject", "general"),
        message=message,
        status="open",
    )
    db.session.add(ticket)
    db.session.commit()

    return jsonify({"ticket": ticket.to_dict()}), 201


# ─────────────────────────────────────────────
# ADMIN — Stats Overview
# ─────────────────────────────────────────────

@admin_bp.route("/stats", methods=["GET"])
@require_admin
def get_stats():
    # #region agent log
    agent_log("E", "admin.py:get_stats", "stats_requested", {
        "has_team_email": bool(getattr(g, "team_admin_email", None)),
    })
    # #endregion
    total_users   = User.query.count()
    premium_users = User.query.filter_by(is_premium=True).count()
    free_users    = total_users - premium_users
    open_tickets  = SupportTicket.query.filter_by(status="open").count()
    total_tickets = SupportTicket.query.count()
    live_premium_jobs = AdminJob.query.filter_by(is_active=True).count()

    return jsonify({
        "total_users":   total_users,
        "premium_users": premium_users,
        "free_users":    free_users,
        "open_tickets":  open_tickets,
        "total_tickets": total_tickets,
        "live_premium_jobs": live_premium_jobs,
    }), 200


# ─────────────────────────────────────────────
# ADMIN — Users
# ─────────────────────────────────────────────

@admin_bp.route("/users", methods=["GET"])
@require_admin
def list_users():
    page     = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    search   = request.args.get("search", "").strip()

    query = User.query.order_by(User.created_at.desc())
    if search:
        like = f"%{search}%"
        query = query.filter(
            db.or_(User.email.ilike(like), User.name.ilike(like))
        )

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    users = [u.to_dict() for u in pagination.items]

    return jsonify({
        "users":       users,
        "total":       pagination.total,
        "pages":       pagination.pages,
        "current_page": page,
    }), 200


@admin_bp.route("/users/<string:user_id>", methods=["DELETE"])
@require_admin
def delete_user(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    db.session.delete(user)
    db.session.commit()
    return jsonify({"message": f"User {user.email} deleted successfully"}), 200


@admin_bp.route("/users/<string:user_id>/premium", methods=["PATCH"])
@require_admin
def toggle_premium(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    data = request.get_json(silent=True) or {}
    user.is_premium   = data.get("is_premium", not user.is_premium)
    user.premium_plan = data.get("premium_plan", "pro-monthly") if user.is_premium else None
    db.session.commit()

    return jsonify({"user": user.to_dict()}), 200


# ─────────────────────────────────────────────
# ADMIN — Support Tickets
# ─────────────────────────────────────────────

@admin_bp.route("/tickets", methods=["GET"])
@require_admin
def list_tickets():
    page     = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    status   = request.args.get("status", "")
    search   = request.args.get("search", "").strip()

    query = SupportTicket.query.order_by(SupportTicket.created_at.desc())
    if status:
        query = query.filter_by(status=status)
    if search:
        like = f"%{search}%"
        query = query.filter(
            db.or_(
                SupportTicket.user_email.ilike(like),
                SupportTicket.ticket_ref.ilike(like),
                SupportTicket.subject.ilike(like),
                SupportTicket.message.ilike(like),
            )
        )

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    tickets = [t.to_dict() for t in pagination.items]

    return jsonify({
        "tickets":      tickets,
        "total":        pagination.total,
        "pages":        pagination.pages,
        "current_page": page,
    }), 200


@admin_bp.route("/tickets/<string:ticket_id>", methods=["PATCH"])
@require_admin
def update_ticket(ticket_id):
    ticket = SupportTicket.query.get(ticket_id)
    if not ticket:
        return jsonify({"error": "Ticket not found"}), 404

    data = request.get_json(silent=True) or {}
    if "status" in data:
        ticket.status = data["status"]
    if "admin_notes" in data:
        ticket.admin_notes = data["admin_notes"]
    ticket.updated_at = datetime.now(timezone.utc)

    db.session.commit()
    return jsonify({"ticket": ticket.to_dict()}), 200


@admin_bp.route("/tickets/<string:ticket_id>", methods=["DELETE"])
@require_admin
def delete_ticket(ticket_id):
    ticket = SupportTicket.query.get(ticket_id)
    if not ticket:
        return jsonify({"error": "Ticket not found"}), 404

    db.session.delete(ticket)
    db.session.commit()
    return jsonify({"message": f"Ticket {ticket.ticket_ref} deleted"}), 200


# ─────────────────────────────────────────────
# Team Admin — Password + Email OTP Login
# ─────────────────────────────────────────────

@admin_bp.route("/team/setup-status", methods=["GET"])
def team_admin_setup_status():
    """Helps debug login issues — only exposes non-secret setup hints."""
    return jsonify({
        "configured": is_team_admin_configured(),
        "expected_email": team_admin_email(),
        "accepted_emails": sorted({team_admin_email(), "team@nyrvexa.in", "team@nirvexa.in"}),
    }), 200


@admin_bp.route("/team/session", methods=["GET"])
@team_admin_required
def team_admin_session():
    """Validate team admin Bearer token."""
    # #region agent log
    agent_log("D", "admin.py:team_admin_session", "session_ok", {"email": g.team_admin_email})
    # #endregion
    return success_response(
        data={"email": g.team_admin_email, "authenticated": True},
        message="Session valid",
    )


@admin_bp.route("/team/resend-otp", methods=["POST"])
@limiter.limit("3 per minute")
def team_admin_resend_otp():
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()

    if not is_team_admin_configured() or not is_allowed_admin_email(email):
        return error_response("Invalid credentials", 401)

    otp = issue_otp(email)
    dev_otp = current_app.config.get("TEAM_ADMIN_DEV_OTP", "")
    skip_email = bool(dev_otp) and current_app.config.get("DEBUG")

    if skip_email:
        sent = True
    else:
        sent = send_team_admin_otp(email, otp)

    if not sent:
        return error_response("Could not send verification email", 503)

    return success_response(
        data={"email": email, "otp_sent": True},
        message="New verification code sent",
    )


@admin_bp.route("/team/login", methods=["POST"])
@limiter.limit("5 per minute")
def team_admin_login():
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    # #region agent log
    agent_log("A", "admin.py:team_admin_login", "login_attempt", {
        "configured": is_team_admin_configured(),
        "email_allowed": is_allowed_admin_email(email),
        "has_jwt_secret": bool(current_app.config.get("JWT_SECRET_KEY")),
        "debug": bool(current_app.config.get("DEBUG")),
        "has_dev_otp": bool(current_app.config.get("TEAM_ADMIN_DEV_OTP")),
    })
    # #endregion

    if not is_team_admin_configured():
        msg = (
            "Team admin password is not configured on the server. "
            "Set TEAM_ADMIN_PASSWORD in backend .env and restart Flask."
        )
        if current_app.config.get("DEBUG"):
            return error_response(msg, 503)
        return error_response("Admin login is temporarily unavailable", 503)

    if not is_allowed_admin_email(email):
        return error_response("Invalid credentials", 401)

    if not verify_password(password):
        # #region agent log
        agent_log("B", "admin.py:team_admin_login", "password_rejected", {"email": email})
        # #endregion
        return error_response("Invalid credentials", 401)

    otp = issue_otp(email)
    dev_otp = current_app.config.get("TEAM_ADMIN_DEV_OTP", "")
    skip_email = bool(dev_otp) and current_app.config.get("DEBUG")

    if skip_email:
        current_app.logger.info("[TeamAdmin] DEV_OTP mode is enabled; email delivery skipped.")
        sent = True
    else:
        sent = send_team_admin_otp(email, otp)

    if not sent:
        if current_app.config.get("DEBUG"):
            return error_response(
                "Could not send OTP email. Set BREVO_API_KEY in .env, or set "
                "TEAM_ADMIN_DEV_OTP=123456 for local testing (development only).",
                503,
            )
        return error_response("Could not send verification email. Try again later.", 503)

    return success_response(
        data={
            "email": email,
            "otp_sent": True,
        },
        message="Verification code sent to your email",
    )


@admin_bp.route("/team/verify-otp", methods=["POST"])
@limiter.limit("10 per minute")
def team_admin_verify_otp():
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()
    otp = data.get("otp", "").strip()

    if not is_allowed_admin_email(email):
        return error_response("Invalid credentials", 401)

    otp_ok = verify_otp(email, otp)
    # #region agent log
    agent_log("C", "admin.py:team_admin_verify_otp", "verify_result", {
        "email": email,
        "otp_len": len(otp),
        "otp_ok": otp_ok,
        "has_jwt_secret": bool(current_app.config.get("JWT_SECRET_KEY")),
    })
    # #endregion

    if not otp_ok:
        return error_response("Invalid or expired verification code", 401)

    if not current_app.config.get("JWT_SECRET_KEY"):
        return error_response(
            "Server misconfigured: set JWT_SECRET_KEY in backend .env and restart.",
            503,
        )

    token = generate_team_admin_token(email)

    forwarded_for = request.headers.get("X-Forwarded-For", "")
    client_ip = (forwarded_for.split(",", 1)[0].strip() if forwarded_for else request.remote_addr) or "Unknown"
    user_agent = request.headers.get("User-Agent", "Unknown")
    alert_email = current_app.config.get("TEAM_ADMIN_ALERT_EMAIL") or team_admin_email()
    alert_sent = send_team_admin_login_alert(
        alert_email=alert_email,
        admin_email=email,
        ip_address=client_ip,
        user_agent=user_agent,
    )
    if not alert_sent:
        current_app.logger.warning("[TeamAdmin] Login alert email failed for %s", email)

    # #region agent log
    agent_log("C", "admin.py:team_admin_verify_otp", "token_issued", {"email": email, "token_len": len(token)})
    # #endregion
    return success_response(
        data={"access_token": token, "email": email},
        message="Signed in successfully",
    )


# ─────────────────────────────────────────────
# Team Admin — Premium Job Posting (form-friendly)
# ─────────────────────────────────────────────

def _parse_skills(raw) -> list:
    if isinstance(raw, list):
        return [str(s).strip() for s in raw if str(s).strip()][:30]
    if isinstance(raw, str) and raw.strip():
        return [s.strip() for s in raw.split(",") if s.strip()][:30]
    return []


def _job_from_payload(data: dict, job: AdminJob | None = None) -> AdminJob:
    """Create: apply all fields. Update: only keys present in JSON (avoids wiping data)."""
    record = job or AdminJob()
    is_update = job is not None

    if not is_update:
        record.title = (data.get("title") or "").strip()
        record.company = (data.get("company") or "").strip()
        record.company_logo = data.get("company_logo")
        record.location = (data.get("location") or "").strip() or None
        record.job_type = (data.get("job_type") or "full-time").strip()
        record.experience = (data.get("experience") or "").strip() or None
        record.salary = (data.get("salary") or "").strip() or None
        record.description = (data.get("description") or "").strip() or None
        record.requirements = (data.get("requirements") or "").strip() or None
        record.skills = _parse_skills(data.get("skills", []))
        record.apply_url = (data.get("apply_url") or "").strip() or None
        record.apply_email = (data.get("apply_email") or "").strip() or None
        record.category = (data.get("category") or "").strip() or None
        record.posted_by = (data.get("posted_by") or "NirVexa Team").strip()
        record.is_active = bool(data.get("is_active", True))
        record.is_featured = bool(data.get("is_featured", False))
        return record

    if "title" in data:
        val = (data.get("title") or "").strip()
        if val:
            record.title = val
    if "company" in data:
        val = (data.get("company") or "").strip()
        if val:
            record.company = val
    if "company_logo" in data:
        record.company_logo = data.get("company_logo")
    if "location" in data:
        record.location = (data.get("location") or "").strip() or None
    if "job_type" in data:
        record.job_type = (data.get("job_type") or "full-time").strip()
    if "experience" in data:
        record.experience = (data.get("experience") or "").strip() or None
    if "salary" in data:
        record.salary = (data.get("salary") or "").strip() or None
    if "description" in data:
        record.description = (data.get("description") or "").strip() or None
    if "requirements" in data:
        record.requirements = (data.get("requirements") or "").strip() or None
    if "skills" in data:
        record.skills = _parse_skills(data.get("skills"))
    if "apply_url" in data:
        record.apply_url = (data.get("apply_url") or "").strip() or None
    if "apply_email" in data:
        record.apply_email = (data.get("apply_email") or "").strip() or None
    if "category" in data:
        record.category = (data.get("category") or "").strip() or None
    if "posted_by" in data:
        record.posted_by = (data.get("posted_by") or "NirVexa Team").strip()
    if "is_active" in data:
        record.is_active = bool(data["is_active"])
    if "is_featured" in data:
        record.is_featured = bool(data["is_featured"])
    return record


@admin_bp.route("/jobs", methods=["GET"])
@team_admin_required
def team_list_jobs():
    jobs = AdminJob.query.order_by(AdminJob.created_at.desc()).all()
    return jsonify({"jobs": [j.to_dict() for j in jobs]}), 200


@admin_bp.route("/jobs", methods=["POST"])
@team_admin_required
def team_create_job():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    company = (data.get("company") or "").strip()

    if not title or not company:
        return error_response("Job title and company name are required", 400)

    job = _job_from_payload(data)
    db.session.add(job)
    db.session.commit()
    current_app.logger.info(
        "[TeamAdmin] Published premium job id=%s title=%r active=%s",
        job.id, job.title, job.is_active,
    )
    return jsonify({"job": job.to_dict()}), 201


@admin_bp.route("/jobs/<string:job_id>", methods=["PATCH"])
@team_admin_required
def team_update_job(job_id):
    job = AdminJob.query.get(job_id)
    if not job:
        return error_response("Job not found", 404)

    data = request.get_json(silent=True) or {}
    job = _job_from_payload(data, job)
    job.updated_at = datetime.now(timezone.utc)
    db.session.commit()
    return jsonify({"job": job.to_dict()}), 200


@admin_bp.route("/jobs/<string:job_id>", methods=["DELETE"])
@team_admin_required
def team_delete_job(job_id):
    job = AdminJob.query.get(job_id)
    if not job:
        return error_response("Job not found", 404)

    db.session.delete(job)
    db.session.commit()
    return jsonify({"message": "Job deleted"}), 200
