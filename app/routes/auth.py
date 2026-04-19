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
                is_verified=True,  # Google emails are verified
            )
            db.session.add(user)
            db.session.flush()

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

    # --- Issue new access token ---
    new_access_token = generate_access_token(user.id)

    return success_response(
        data={"access_token": new_access_token},
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

    return success_response(
        data={"user": user.to_dict()},
        message="User profile retrieved"
    )