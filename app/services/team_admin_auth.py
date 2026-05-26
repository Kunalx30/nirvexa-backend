"""Team admin login — password + email OTP for team@nyrvexa.in."""

import os
import random
import secrets
import threading
from datetime import datetime, timedelta, timezone

import jwt
from flask import current_app
from flask_bcrypt import Bcrypt
from werkzeug.security import check_password_hash as werkzeug_check_hash

_bcrypt = Bcrypt()

_otp_lock = threading.Lock()
_otp_store: dict[str, dict] = {}

# Accepted login emails (primary + common spelling variants)
_EMAIL_ALIASES = frozenset({
    "team@nyrvexa.in",
    "team@nirvexa.in",
})


def _config(key: str, default: str = "") -> str:
    """Read from Flask config when in app context, else os.environ."""
    try:
        if current_app:
            val = current_app.config.get(key)
            if val is not None:
                return str(val).strip()
    except RuntimeError:
        pass
    return os.getenv(key, default).strip()


def team_admin_email() -> str:
    return _config("TEAM_ADMIN_EMAIL", "team@nyrvexa.in").lower()


def team_admin_password() -> str:
    return _config("TEAM_ADMIN_PASSWORD", "")


def is_team_admin_configured() -> bool:
    return bool(team_admin_password())


def is_allowed_admin_email(email: str) -> bool:
    normalized = (email or "").strip().lower()
    if normalized in _EMAIL_ALIASES:
        return True
    return normalized == team_admin_email()


def verify_password(plain: str) -> bool:
    stored = team_admin_password()
    if not stored:
        return False

    plain = (plain or "").strip()
    if not plain:
        return False

    # Strip accidental quotes from .env copy-paste
    if (stored.startswith('"') and stored.endswith('"')) or (
        stored.startswith("'") and stored.endswith("'")
    ):
        stored = stored[1:-1]

    if stored.startswith("$2"):
        try:
            return _bcrypt.check_password_hash(stored, plain)
        except Exception:
            return werkzeug_check_hash(stored, plain)

    return secrets.compare_digest(stored, plain)


def issue_otp(email: str) -> str:
    dev_otp = _config("TEAM_ADMIN_DEV_OTP", "")
    try:
        debug_mode = bool(current_app.config.get("DEBUG"))
    except RuntimeError:
        debug_mode = os.getenv("FLASK_ENV", "development") == "development"

    if dev_otp and debug_mode:
        code = dev_otp
    else:
        code = f"{random.randint(100000, 999999)}"

    expires = datetime.now(timezone.utc) + timedelta(minutes=10)
    with _otp_lock:
        _otp_store[email] = {"code": code, "expires": expires, "attempts": 0}
    return code


def verify_otp(email: str, code: str) -> bool:
    code = str(code).strip()
    dev_otp = _config("TEAM_ADMIN_DEV_OTP", "")
    try:
        debug_mode = bool(current_app.config.get("DEBUG"))
    except RuntimeError:
        debug_mode = os.getenv("FLASK_ENV", "development") == "development"

    # Dev: accept fixed OTP even if store was cleared (restart) or code was already used
    if dev_otp and debug_mode and code == dev_otp and is_allowed_admin_email(email):
        with _otp_lock:
            _otp_store.pop(email, None)
        # #region agent log
        from app.utils.debug_log import agent_log
        agent_log("C", "team_admin_auth.py:verify_otp", "dev_otp_bypass", {"email": email, "runId": "post-fix"})
        # #endregion
        return True

    with _otp_lock:
        entry = _otp_store.get(email)
        if not entry:
            return False
        if entry["attempts"] >= 5:
            return False
        entry["attempts"] += 1
        if datetime.now(timezone.utc) > entry["expires"]:
            _otp_store.pop(email, None)
            return False
        if entry["code"] != str(code).strip():
            return False
        _otp_store.pop(email, None)
        return True


def generate_team_admin_token(email: str) -> str:
    payload = {
        "sub": email,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=8),
        "type": "team_admin",
    }
    token = jwt.encode(
        payload,
        current_app.config["JWT_SECRET_KEY"],
        algorithm="HS256",
    )
    return token if isinstance(token, str) else token.decode("utf-8")


def decode_team_admin_token(token: str) -> dict:
    payload = jwt.decode(
        token,
        current_app.config["JWT_SECRET_KEY"],
        algorithms=["HS256"],
    )
    if payload.get("type") != "team_admin":
        raise jwt.InvalidTokenError("Not a team admin token")
    return payload
