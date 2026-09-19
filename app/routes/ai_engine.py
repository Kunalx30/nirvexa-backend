"""
app/routes/ai_engine.py
Flask Blueprint for Nirvexa AI Engine Phase 1 endpoints.
"""
import logging
from flask import Blueprint, request, jsonify, current_app
from pydantic import ValidationError

from app.ai_engine.coordinator import WebResearchEngine
from app.ai_engine.schemas.research import SearchRequest
from app.middleware.rate_limiter import limiter

logger = logging.getLogger(__name__)

ai_engine_bp = Blueprint("ai_engine", __name__)

# Module-level engine singleton.
# SearchService reads AI_ENGINE_SEARCH_PROVIDER from the environment at instantiation.
# This is acceptable because the env var is set before the module is imported.
_engine = WebResearchEngine()

# Maximum acceptable request body size (bytes) — prevents resource exhaustion
# from huge JSON payloads before Pydantic validation runs.
_MAX_REQUEST_BYTES = 4096  # 4 KB is far more than any valid SearchRequest


@ai_engine_bp.route("/research/search", methods=["POST"])
@limiter.limit("10 per minute")
def research_search():
    """
    POST /api/ai/research/search
    Rate-limited: 10 requests per IP per minute.
    Accepts: { "query": str, "max_results": Optional[int], "fetch_content": Optional[bool] }
    Returns: ResearchResponse JSON with structured search results and extracted webpage content.
    """
    # 1. Feature flag guard
    if not current_app.config.get("AI_ENGINE_ENABLED", True):
        return jsonify({
            "error": "ai_engine_disabled",
            "message": "AI Engine web research is currently disabled.",
        }), 503

    # 2. Enforce request body size limit before any parsing
    content_length = request.content_length
    if content_length is not None and content_length > _MAX_REQUEST_BYTES:
        return jsonify({
            "error": "request_too_large",
            "message": f"Request body must not exceed {_MAX_REQUEST_BYTES} bytes.",
        }), 413

    # 3. Extract JSON body
    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict) or not data:
        return jsonify({
            "error": "validation_error",
            "message": "Request body must be a non-empty JSON object containing 'query'.",
        }), 400

    # 4. Validate request schema
    try:
        search_req = SearchRequest(**data)
    except ValidationError as e:
        clean_errors = [
            {"field": str(err.get("loc", [""])[-1]), "message": str(err.get("msg", ""))}
            for err in e.errors()
        ]
        return jsonify({
            "error": "validation_error",
            "message": "; ".join(err["message"] for err in clean_errors),
            "details": clean_errors,
        }), 400
    except (ValueError, TypeError) as e:
        return jsonify({
            "error": "validation_error",
            "message": str(e),
        }), 400

    # 5. Execute research pipeline
    try:
        response = _engine.research(search_req)
        # Use model_dump if Pydantic v2, fallback to dict() for v1
        if hasattr(response, "model_dump"):
            payload = response.model_dump()
        else:
            payload = response.dict()

        # Surface provider errors as 503 to the caller instead of silent 200+empty
        if response.search_status == "provider_error":
            return jsonify(payload), 503

        return jsonify(payload), 200

    except Exception as e:
        logger.error(
            "[AI Engine] Unhandled error for query=%r: %s",
            getattr(search_req, "query", "unknown"),
            e,
            exc_info=True,
        )
        return jsonify({
            "error": "research_failed",
            "message": "An error occurred while executing web research.",
        }), 500
