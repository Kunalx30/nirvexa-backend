from datetime import datetime, timezone
from flask import Blueprint, request, g

from app.database.db import db
from app.models.chat_message import ChatMessage
from app.models.chat_session import ChatSession
from app.middleware.auth_middleware import token_required
from app.middleware.rate_limiter import limiter
from app.services.llm_router import route_and_call
from app.utils.helpers import success_response, error_response


chat_bp = Blueprint("chat", __name__)


def _default_title(message: str) -> str:
    title = " ".join((message or "").strip().split())
    return title[:57] + "..." if len(title) > 60 else title or "New chat"


def _get_or_create_session(user_id: str, session_id: str | None, first_message: str):
    if session_id:
        return ChatSession.query.filter_by(id=session_id, user_id=user_id).first()

    session = (
        ChatSession.query
        .filter_by(user_id=user_id)
        .order_by(ChatSession.updated_at.desc())
        .first()
    )
    if session:
        return session

    session = ChatSession(user_id=user_id, title=_default_title(first_message))
    db.session.add(session)
    db.session.flush()
    return session


@chat_bp.route("/chat/sessions", methods=["POST"])
@token_required
@limiter.limit("20 per minute")
def create_chat_session():
    """Create a new chat thread for the current user."""
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "New chat").strip()[:120] or "New chat"

    session = ChatSession(user_id=g.user_id, title=title)
    db.session.add(session)
    db.session.commit()

    return success_response(
        data={"session": session.to_dict()},
        message="Chat session created",
        status_code=201,
    )


@chat_bp.route("/chat/sessions", methods=["GET"])
@token_required
@limiter.limit("60 per minute")
def list_chat_sessions():
    """List the current user's chat threads for a sidebar."""
    sessions = (
        ChatSession.query
        .filter_by(user_id=g.user_id)
        .order_by(ChatSession.updated_at.desc())
        .all()
    )

    return success_response(
        data={"sessions": [session.to_dict() for session in sessions]},
        message="Chat sessions retrieved",
    )


@chat_bp.route("/chat/sessions/<session_id>", methods=["PATCH"])
@token_required
@limiter.limit("30 per minute")
def update_chat_session(session_id):
    """Rename a chat thread."""
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return error_response("Title cannot be empty", 400)

    session = ChatSession.query.filter_by(id=session_id, user_id=g.user_id).first()
    if not session:
        return error_response("Chat session not found", 404)

    session.title = title[:120]
    session.touch()
    db.session.commit()

    return success_response(
        data={"session": session.to_dict()},
        message="Chat session updated",
    )


@chat_bp.route("/chat/sessions/<session_id>", methods=["DELETE"])
@token_required
@limiter.limit("20 per minute")
def delete_chat_session(session_id):
    """Delete a chat thread and its messages."""
    session = ChatSession.query.filter_by(id=session_id, user_id=g.user_id).first()
    if not session:
        return error_response("Chat session not found", 404)

    db.session.delete(session)
    db.session.commit()

    return success_response(
        data={"session_id": session_id},
        message="Chat session deleted",
    )


@chat_bp.route("/chat", methods=["POST"])
@token_required
@limiter.limit("30 per minute")
def chat():
    """
    Main AI chat endpoint.
    Accepts optional session_id to continue a specific chat thread.
    """
    data = request.get_json(silent=True)
    if not data:
        return error_response("Request body is required", 400)

    user_message = data.get("message", "").strip()
    if not user_message:
        return error_response("Message cannot be empty", 400)

    if len(user_message) > 4000:
        return error_response("Message is too long - maximum 4000 characters", 400)

    user_id = g.user_id
    session = _get_or_create_session(user_id, data.get("session_id"), user_message)
    if not session:
        return error_response("Chat session not found", 404)

    history_records = (
        ChatMessage.query
        .filter_by(user_id=user_id, session_id=session.id)
        .order_by(ChatMessage.created_at.desc())
        .limit(10)
        .all()
    )
    history_records.reverse()

    conversation_history = [
        {"role": msg.role, "content": msg.content}
        for msg in history_records
    ]

    result = route_and_call(
        user_message=user_message,
        conversation_history=conversation_history,
        max_tokens=int(data.get("max_tokens", 1024)),
        temperature=float(data.get("temperature", 0.7)),
    )

    ai_response = result["response"]
    model_used = result["model_used"]

    if session.title == "New chat":
        session.title = _default_title(user_message)
    session.updated_at = datetime.now(timezone.utc)

    user_msg = ChatMessage(
        user_id=user_id,
        session_id=session.id,
        role="user",
        content=user_message,
        model_used=None,
    )
    db.session.add(user_msg)

    ai_msg = ChatMessage(
        user_id=user_id,
        session_id=session.id,
        role="assistant",
        content=ai_response,
        model_used=model_used,
    )
    db.session.add(ai_msg)
    db.session.commit()

    return success_response(
        data={
            "session": session.to_dict(),
            "session_id": session.id,
            "message": ai_response,
            "model_used": model_used,
            "provider_used": result["provider_used"],
            "intent": result["intent"],
            "used_fallback": result["used_fallback"],
        },
        message="Response generated successfully",
    )


@chat_bp.route("/chat/history", methods=["GET"])
@token_required
@limiter.limit("60 per minute")
def get_chat_history():
    """
    Get paginated chat history.
    Optional query param: session_id=<chat session id>
    """
    user_id = g.user_id
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 20, type=int), 50)
    session_id = request.args.get("session_id")

    query = ChatMessage.query.filter_by(user_id=user_id)
    if session_id:
        session = ChatSession.query.filter_by(id=session_id, user_id=user_id).first()
        if not session:
            return error_response("Chat session not found", 404)
        query = query.filter_by(session_id=session_id)

    messages = query.order_by(ChatMessage.created_at.desc()).paginate(
        page=page,
        per_page=per_page,
        error_out=False,
    )

    return success_response(
        data={
            "session_id": session_id,
            "messages": [m.to_dict() for m in messages.items],
            "total": messages.total,
            "page": messages.page,
            "per_page": per_page,
            "has_more": messages.has_next,
        },
        message="Chat history retrieved",
    )


@chat_bp.route("/chat/history", methods=["DELETE"])
@token_required
@limiter.limit("10 per minute")
def clear_chat_history():
    """Clear all chat history or only one chat session's history."""
    user_id = g.user_id
    session_id = request.args.get("session_id")

    query = ChatMessage.query.filter_by(user_id=user_id)
    if session_id:
        session = ChatSession.query.filter_by(id=session_id, user_id=user_id).first()
        if not session:
            return error_response("Chat session not found", 404)
        query = query.filter_by(session_id=session_id)

    deleted = query.delete()
    db.session.commit()

    return success_response(
        data={"session_id": session_id, "deleted_count": deleted},
        message="Chat history cleared successfully",
    )
