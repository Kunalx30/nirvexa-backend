import jwt
import uuid
from datetime import datetime, timezone
from flask import current_app


def generate_access_token(user_id: str) -> str:
    """Generate a short-lived JWT access token."""
    payload = {
        "sub": user_id,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + current_app.config["JWT_ACCESS_TOKEN_EXPIRES"],
        "type": "access",
        "jti": str(uuid.uuid4()),  # Unique token ID
    }
    return jwt.encode(
        payload,
        current_app.config["JWT_SECRET_KEY"],
        algorithm="HS256"
    )


def generate_refresh_token(user_id: str) -> str:
    """Generate a long-lived JWT refresh token."""
    payload = {
        "sub": user_id,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + current_app.config["JWT_REFRESH_TOKEN_EXPIRES"],
        "type": "refresh",
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(
        payload,
        current_app.config["JWT_SECRET_KEY"],
        algorithm="HS256"
    )


def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT token.
    Returns payload dict or raises jwt exceptions.
    """
    return jwt.decode(
        token,
        current_app.config["JWT_SECRET_KEY"],
        algorithms=["HS256"]
    )


def success_response(data: dict, message: str = "Success", status_code: int = 200):
    """Standard success response format across all endpoints."""
    from flask import jsonify
    return jsonify({
        "success": True,
        "message": message,
        "data": data,
    }), status_code


def error_response(message: str, status_code: int, errors: list = None):
    """Standard error response format across all endpoints."""
    from flask import jsonify
    response = {
        "success": False,
        "message": message,
    }
    if errors:
        response["errors"] = errors
    return jsonify(response), status_code