import jwt
from functools import wraps
from flask import request, g
from app.utils.helpers import decode_token, error_response


def token_required(f):
    """
    JWT Authentication Decorator.
    Protects any route — attach with @token_required.
    Injects the authenticated user_id into Flask's g object.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]

        if not token:
            return error_response("Authentication token is missing", 401)

        try:
            payload = decode_token(token)

            # Make sure it's an access token, not a refresh token
            if payload.get("type") != "access":
                return error_response("Invalid token type", 401)

            # Attach user_id to request context
            g.user_id = payload["sub"]

        except jwt.ExpiredSignatureError:
            return error_response("Token has expired. Please log in again.", 401)
        except jwt.InvalidTokenError:
            return error_response("Invalid token. Please log in again.", 401)

        return f(*args, **kwargs)

    return decorated