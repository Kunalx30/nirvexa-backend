import jwt
from functools import wraps
from flask import request, g

from app.services.team_admin_auth import decode_team_admin_token
from app.utils.helpers import error_response
from app.utils.debug_log import agent_log


def team_admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return error_response("Team admin authentication required", 401)

        token = auth_header.split(" ", 1)[1]
        try:
            payload = decode_team_admin_token(token)
            g.team_admin_email = payload["sub"]
        except jwt.ExpiredSignatureError:
            # #region agent log
            agent_log("D", "team_admin_middleware.py", "token_expired", {})
            # #endregion
            return error_response("Session expired. Please sign in again.", 401)
        except jwt.InvalidTokenError as exc:
            # #region agent log
            agent_log("D", "team_admin_middleware.py", "token_invalid", {"reason": type(exc).__name__})
            # #endregion
            return error_response("Invalid team admin session", 401)

        return f(*args, **kwargs)

    return decorated
