from flask import Blueprint, request, g
from app.database.db import db
from app.models.chat_message import ChatMessage
from app.middleware.auth_middleware import token_required
from app.middleware.rate_limiter import limiter
from app.services.llm_router import route_and_call
from app.utils.helpers import success_response, error_response

chat_bp = Blueprint("chat", __name__)


# ─────────────────────────────────────────────
# POST /api/chat
# ─────────────────────────────────────────────
@chat_bp.route("/chat", methods=["POST"])
@token_required
@limiter.limit("30 per minute")
def chat():
    """
    Main AI chat endpoint.
    - Loads last 10 messages from DB for context memory
    - Routes to best available LLM based on intent
    - Saves both user message and AI reply to DB
    - Returns AI response with model info
    """
    data = request.get_json()

    if not data:
        return error_response("Request body is required", 400)

    user_message = data.get("message", "").strip()
    if not user_message:
        return error_response("Message cannot be empty", 400)

    if len(user_message) > 4000:
        return error_response("Message is too long — maximum 4000 characters", 400)

    user_id = g.user_id

    # --- Load conversation history from DB ---
    history_records = (
        ChatMessage.query
        .filter_by(user_id=user_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(10)
        .all()
    )

    # Reverse to get chronological order
    history_records.reverse()

    # Format for LLM — list of {role, content}
    conversation_history = [
        {"role": msg.role, "content": msg.content}
        for msg in history_records
    ]

    # --- Route to best LLM ---
    result = route_and_call(
        user_message=user_message,
        conversation_history=conversation_history,
        max_tokens=int(data.get("max_tokens", 1024)),
        temperature=float(data.get("temperature", 0.7)),
    )

    ai_response = result["response"]
    model_used  = result["model_used"]

    # --- Save user message to DB ---
    user_msg = ChatMessage(
        user_id=user_id,
        role="user",
        content=user_message,
        model_used=None,
    )
    db.session.add(user_msg)

    # --- Save AI reply to DB ---
    ai_msg = ChatMessage(
        user_id=user_id,
        role="assistant",
        content=ai_response,
        model_used=model_used,
    )
    db.session.add(ai_msg)
    db.session.commit()

    return success_response(
        data={
            "message":       ai_response,
            "model_used":    model_used,
            "provider_used": result["provider_used"],
            "intent":        result["intent"],
            "used_fallback": result["used_fallback"],
        },
        message="Response generated successfully"
    )


# ─────────────────────────────────────────────
# GET /api/chat/history
# ─────────────────────────────────────────────
@chat_bp.route("/chat/history", methods=["GET"])
@token_required
@limiter.limit("60 per minute")
def get_chat_history():
    """
    Get paginated chat history for the current user.
    Query params: page (default 1), per_page (default 20)
    """
    user_id  = g.user_id
    page     = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 20, type=int), 50)

    messages = (
        ChatMessage.query
        .filter_by(user_id=user_id)
        .order_by(ChatMessage.created_at.desc())
        .paginate(page=page, per_page=per_page, error_out=False)
    )

    return success_response(
        data={
            "messages":   [m.to_dict() for m in messages.items],
            "total":      messages.total,
            "page":       messages.page,
            "per_page":   per_page,
            "has_more":   messages.has_next,
        },
        message="Chat history retrieved"
    )


# ─────────────────────────────────────────────
# DELETE /api/chat/history
# ─────────────────────────────────────────────
@chat_bp.route("/chat/history", methods=["DELETE"])
@token_required
@limiter.limit("10 per minute")
def clear_chat_history():
    """Clear all chat history for the current user."""
    user_id = g.user_id

    deleted = ChatMessage.query.filter_by(user_id=user_id).delete()
    db.session.commit()

    return success_response(
        data={"deleted_count": deleted},
        message="Chat history cleared successfully"
    )