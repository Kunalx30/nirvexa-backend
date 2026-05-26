import requests
from flask import Blueprint, request, g, current_app
from flask_bcrypt import Bcrypt
from datetime import datetime, timezone

from app.database.db import db
from app.models.user import User
from app.middleware.auth_middleware import token_required
from app.middleware.rate_limiter import limiter
from app.utils.validators import validate_email, validate_password, validate_name
from app.utils.helpers import (
    generate_access_token,
    generate_refresh_token,
    decode_token,
    success_response,
    error_response,
)

import secrets
from datetime import timedelta

import logging
logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__)
bcrypt = Bcrypt()


# ─────────────────────────────────────────────
# POST /api/auth/register
# ─────────────────────────────────────────────
@auth_bp.route("/register", methods=["POST"])
@limiter.limit("5 per minute")
def register():
    """Register a new user with email and password."""
    data = request.get_json()

    # --- Validate input presence ---
    if not data:
        return error_response("Request body is required", 400)

    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not all([name, email, password]):
        return error_response("Name, email, and password are required", 400)

    # --- Validate formats ---
    if not validate_name(name):
        return error_response("Name must be 2-100 characters, letters only", 400)

    if not validate_email(email):
        return error_response("Invalid email address format", 400)

    password_check = validate_password(password)
    if not password_check["is_valid"]:
        return error_response(
            "Password does not meet requirements",
            400,
            errors=password_check["errors"]
        )

    # --- Check if email already exists ---
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return error_response("An account with this email already exists", 409)

    # --- Hash password and create user ---
    password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    new_user = User(
        name=name,
        email=email,
        password_hash=password_hash,
        is_verified=False,
    )
    db.session.add(new_user)
    db.session.flush()  # Get the ID before commit

    # --- Generate tokens ---
    access_token = generate_access_token(new_user.id)
    refresh_token = generate_refresh_token(new_user.id)
    new_user.refresh_token = refresh_token
    new_user.update_last_login()

    db.session.commit()


    # Send welcome email
    from app.services.email_service import send_welcome_email
    send_welcome_email(new_user.email, new_user.name)

    return success_response(
        data={
            "user": new_user.to_dict(),
            "access_token": access_token,
            "refresh_token": refresh_token,
        },
        message="Account created successfully",
        status_code=201
    )


# ─────────────────────────────────────────────
# POST /api/auth/login
# ─────────────────────────────────────────────
@auth_bp.route("/login", methods=["POST"])
@limiter.limit("10 per minute")
def login():
    """Login with email and password."""
    data = request.get_json()

    if not data:
        return error_response("Request body is required", 400)

    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not all([email, password]):
        return error_response("Email and password are required", 400)

    # --- Find user ---
    user = User.query.filter_by(email=email).first()

    # Deliberate vague message — don't reveal if email exists
    if not user or not user.password_hash:
        return error_response("Invalid email or password", 401)

    if not user.is_active:
        return error_response("This account has been deactivated", 403)

    # --- Verify password ---
    if not bcrypt.check_password_hash(user.password_hash, password):
        return error_response("Invalid email or password", 401)

    # --- Generate new tokens ---
    access_token = generate_access_token(user.id)
    refresh_token = generate_refresh_token(user.id)
    user.refresh_token = refresh_token
    user.update_last_login()

    db.session.commit()

    return success_response(
        data={
            "user": user.to_dict(),
            "access_token": access_token,
            "refresh_token": refresh_token,
        },
        message="Login successful"
    )


# ─────────────────────────────────────────────
# POST /api/auth/google
# ─────────────────────────────────────────────
@auth_bp.route("/google", methods=["POST"])
@limiter.limit("10 per minute")
def google_login():
    """Login or register using Google OAuth ID token."""
    data = request.get_json()

    if not data:
        return error_response("Request body is required", 400)

    id_token = (data.get("idToken") or data.get("id_token") or "").strip()
    if not id_token:
        return error_response("Google ID token is required", 400)

    # --- Verify token with Google ---
    try:
        google_response = requests.get(
            f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}",
            timeout=10
        )
        if google_response.status_code != 200:
            return error_response("Failed to verify Google token", 401)

        google_data = google_response.json()

        # Verify the token was issued for our app
        if google_data.get("aud") != current_app.config["GOOGLE_CLIENT_ID"]:
            return error_response("Google token was not issued for this app", 401)

    except requests.RequestException:
        return error_response("Could not connect to Google verification service", 503)

    google_id = google_data.get("sub")
    email = google_data.get("email", "").lower()
    name = google_data.get("name", "")
    avatar_url = google_data.get("picture", "")

    if not google_id or not email:
        return error_response("Could not extract user info from Google token", 400)

    # --- Find or create user ---
    user = User.query.filter_by(google_id=google_id).first()

    if not user:
        # Check if email already exists (user registered with email before)
        user = User.query.filter_by(email=email).first()
        if user:
            # Link Google account to existing email account
            user.google_id = google_id
            user.avatar_url = avatar_url or user.avatar_url
        else:
            # Brand new user via Google
            user = User(
                name=name,
                email=email,
                google_id=google_id,
                avatar_url=avatar_url,
                is_verified=True,
            )
            db.session.add(user)
            db.session.flush()

            # Send welcome email for new Google signups
            from app.services.email_service import send_welcome_email
            send_welcome_email(user.email, user.name)
    if not user.is_active:
        return error_response("This account has been deactivated", 403)

    # --- Generate tokens ---
    access_token = generate_access_token(user.id)
    refresh_token = generate_refresh_token(user.id)
    user.refresh_token = refresh_token
    user.update_last_login()

    db.session.commit()

    return success_response(
        data={
            "user": user.to_dict(),
            "access_token": access_token,
            "refresh_token": refresh_token,
        },
        message="Google login successful"
    )


# ─────────────────────────────────────────────
# POST /api/auth/refresh
# ─────────────────────────────────────────────
@auth_bp.route("/refresh", methods=["POST"])
@limiter.limit("20 per minute")
def refresh():
    """Exchange a valid refresh token for a new access token."""
    data = request.get_json()

    if not data:
        return error_response("Request body is required", 400)

    refresh_token = data.get("refresh_token", "").strip()
    if not refresh_token:
        return error_response("Refresh token is required", 400)

    try:
        payload = decode_token(refresh_token)

        if payload.get("type") != "refresh":
            return error_response("Invalid token type", 401)

        user_id = payload["sub"]

    except Exception:
        return error_response("Invalid or expired refresh token", 401)

    # --- Verify token matches what's stored in DB ---
    user = User.query.filter_by(id=user_id).first()

    if not user or user.refresh_token != refresh_token:
        return error_response("Refresh token is invalid or has been revoked", 401)

    if not user.is_active:
        return error_response("This account has been deactivated", 403)

    # --- Issue new access token + fresh profile (premium expiry synced) ---
    from app.middleware.rate_limiter import _is_premium_active

    _is_premium_active(user)
    new_access_token = generate_access_token(user.id)

    return success_response(
        data={"access_token": new_access_token, "user": user.to_dict()},
        message="Access token refreshed successfully"
    )


# ─────────────────────────────────────────────
# POST /api/auth/logout
# ─────────────────────────────────────────────
@auth_bp.route("/logout", methods=["POST"])
@token_required
def logout():
    """Logout — invalidate the refresh token in the database."""
    user = User.query.filter_by(id=g.user_id).first()

    if user:
        user.refresh_token = None
        db.session.commit()

    return success_response(
        data={},
        message="Logged out successfully"
    )


# ─────────────────────────────────────────────
# GET /api/auth/me
# ─────────────────────────────────────────────
@auth_bp.route("/me", methods=["GET"])
@token_required
def get_current_user():
    """Get the currently authenticated user's profile."""
    user = User.query.filter_by(id=g.user_id).first()

    if not user:
        return error_response("User not found", 404)

    from app.middleware.rate_limiter import _is_premium_active

    _is_premium_active(user)

    return success_response(
        data={"user": user.to_dict()},
        message="User profile retrieved"
    )


# ─────────────────────────────────────────────
# PUT /api/auth/me
# ─────────────────────────────────────────────
@auth_bp.route("/me", methods=["PUT"])
@token_required
def update_profile():
    """Update skills, preferred_location, job_type, experience_level."""
    data = request.get_json()

    if not data:
        return error_response("Request body is required", 400)

    user = User.query.filter_by(id=g.user_id).first()
    if not user:
        return error_response("User not found", 404)

    if "skills" in data:
        skills = data["skills"]
        if isinstance(skills, list):
            user.skills = [s.strip() for s in skills if isinstance(s, str) and s.strip()]

    if "name" in data:
        name = (data["name"] or "").strip()
        if name:
            user.name = name

    if "avatar_url" in data:
        user.avatar_url = (data["avatar_url"] or "").strip() or None

    if "preferred_location" in data:
        user.preferred_location = (data["preferred_location"] or "").strip() or None

    if "job_type" in data:
        valid_types = ["full-time", "part-time", "internship", "freelance", "remote"]
        if data["job_type"] in valid_types:
            user.job_type = data["job_type"]

    if "experience_level" in data:
        valid_levels = ["fresher", "junior", "mid", "senior", "lead"]
        if data["experience_level"] in valid_levels:
            user.experience_level = data["experience_level"]

    db.session.commit()

    return success_response(
        data={"user": user.to_dict()},
        message="Profile updated successfully"
    )


# ─────────────────────────────────────────────
# PUT /api/auth/change-password
# ─────────────────────────────────────────────
@auth_bp.route("/change-password", methods=["PUT"])
@token_required
def change_password():
    """Change password — requires current password verification."""
    data = request.get_json()

    if not data:
        return error_response("Request body is required", 400)

    current_password = data.get("current_password", "")
    new_password = data.get("new_password", "")

    if not current_password or not new_password:
        return error_response("current_password and new_password are required", 400)

    user = User.query.filter_by(id=g.user_id).first()
    if not user:
        return error_response("User not found", 404)

    if not user.password_hash:
        return error_response("Google accounts cannot change password here", 400)

    if not bcrypt.check_password_hash(user.password_hash, current_password):
        return error_response("Current password is incorrect", 401)

    password_check = validate_password(new_password)
    if not password_check["is_valid"]:
        return error_response(
            "Password does not meet requirements",
            400,
            errors=password_check["errors"]
        )

    user.password_hash = bcrypt.generate_password_hash(new_password).decode("utf-8")
    user.refresh_token = None  # force re-login on all devices
    db.session.commit()

    return success_response(data={}, message="Password changed successfully")

# ─────────────────────────────────────────────
# POST /api/auth/forgot-password
# ─────────────────────────────────────────────
@auth_bp.route("/forgot-password", methods=["POST"])
@limiter.limit("3 per minute")
def forgot_password():
    data = request.get_json()
    if not data:
        return error_response("Request body is required", 400)

    email = (data.get("email") or "").strip().lower()
    if not email:
        return error_response("Email is required", 400)
    

    user = User.query.filter_by(email=email).first()

    # Always return success — don't reveal if email exists
    if user and user.is_active:
        token = secrets.token_urlsafe(32)
        user.reset_token = token
        user.reset_token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)
        db.session.commit()

        frontend_url = current_app.config.get("FRONTEND_URL", "http://localhost:5173")
        reset_url = f"{frontend_url}/reset-password?token={token}"
        logger.warning(f"[DEV] Reset URL: {reset_url}")

        from app.services.email_service import send_password_reset
        send_password_reset(user.email, user.name, reset_url)

    return success_response(
        data={},
        message="If this email exists, a reset link has been sent."
    )


# ─────────────────────────────────────────────
# POST /api/auth/reset-password
# ─────────────────────────────────────────────
@auth_bp.route("/reset-password", methods=["POST"])
def reset_password():
    data = request.get_json()
    if not data:
        return error_response("Request body is required", 400)

    token       = (data.get("token") or "").strip()
    new_password = (data.get("new_password") or "")

    if not token or not new_password:
        return error_response("token and new_password are required", 400)

    user = User.query.filter_by(reset_token=token).first()

    if not user or not user.reset_token_expiry:
        return error_response("Invalid or expired reset link", 400)

    if datetime.now(timezone.utc) > user.reset_token_expiry:
        return error_response("Reset link has expired. Please request a new one.", 400)

    password_check = validate_password(new_password)
    if not password_check["is_valid"]:
        return error_response("Password does not meet requirements", 400,
                              errors=password_check["errors"])

    user.password_hash     = bcrypt.generate_password_hash(new_password).decode("utf-8")
    user.reset_token       = None
    user.reset_token_expiry = None
    user.refresh_token     = None  # force re-login everywhere
    db.session.commit()

    return success_response(data={}, message="Password reset successfully. Please log in.")