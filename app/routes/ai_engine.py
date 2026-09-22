"""
app/routes/ai_engine.py
Flask Blueprint for Nirvexa AI Engine Phase 1 & 2 endpoints.
"""
import logging
from flask import Blueprint, request, jsonify, current_app, g
from pydantic import ValidationError

from app.ai_engine.coordinator import WebResearchEngine
from app.ai_engine.documents.parser import DocumentParserError
from app.ai_engine.documents.service import QuotaExceededError
from app.ai_engine.reasoning.schemas import AnswerRequest
from app.ai_engine.retrieval.schemas import EvidenceRequest
from app.ai_engine.schemas.research import SearchRequest
from app.middleware.auth_middleware import token_required
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
@token_required
@limiter.limit("10 per minute")
def research_search():
    """
    POST /api/ai/research/search
    Authenticated: requires valid JWT Bearer access token.
    Rate-limited: 10 requests per IP per minute.
    Accepts: { "query": str, "max_results": Optional[int], "fetch_content": Optional[bool] }
    Returns: ResearchResponse JSON with structured search results and extracted webpage content.
    """
    # 1. Feature flag guard
    if not current_app.config.get("AI_ENGINE_ENABLED", False):
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


@ai_engine_bp.route("/research/evidence", methods=["POST"])
@token_required
@limiter.limit("10 per minute")
def research_evidence():
    """
    POST /api/ai/research/evidence
    Internal/Development endpoint: Transforms web research into ranked, budgeted EvidencePack.
    Authenticated: requires valid JWT Bearer access token.
    Rate-limited: 10 requests per IP per minute.
    Accepts: { "query": str, "max_results": Optional[int], "max_evidence_items": Optional[int], "max_evidence_chars": Optional[int] }
    Returns: EvidencePack JSON with ranked, deduplicated EvidenceItem chunks and source summaries.
    """
    # 1. Feature flag guard
    if not current_app.config.get("AI_ENGINE_ENABLED", False):
        return jsonify({
            "error": "ai_engine_disabled",
            "message": "AI Engine web research is currently disabled.",
        }), 503

    # 2. Enforce request body size limit
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
        evidence_req = EvidenceRequest(**data)
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

    # 5. Check document feature flag if search_mode uses documents
    if getattr(evidence_req, "search_mode", "web") in ("document", "hybrid"):
        if not current_app.config.get("AI_ENGINE_DOCUMENTS_ENABLED", True):
            return jsonify({
                "error": "ai_engine_documents_disabled",
                "message": "Document retrieval is currently disabled.",
            }), 503

    # 6. Execute research + evidence pipeline
    try:
        search_req = SearchRequest(
            query=evidence_req.query,
            max_results=evidence_req.max_results or 5,
            fetch_content=True,
        )
        evidence_pack = _engine.research_evidence(
            search_req,
            max_evidence_items=evidence_req.max_evidence_items,
            max_evidence_chars=evidence_req.max_evidence_chars,
            search_mode=getattr(evidence_req, "search_mode", "web"),
            user_id=getattr(g, "user_id", None),
        )
        if hasattr(evidence_pack, "model_dump"):
            payload = evidence_pack.model_dump()
        else:
            payload = evidence_pack.dict()

        return jsonify(payload), 200

    except Exception as e:
        logger.error(
            "[AI Engine] Unhandled evidence pipeline error for query=%r: %s",
            getattr(evidence_req, "query", "unknown"),
            e,
            exc_info=True,
        )
        return jsonify({
            "error": "evidence_failed",
            "message": "An error occurred while building evidence pack.",
        }), 500


@ai_engine_bp.route("/research/answer", methods=["POST"])
@token_required
@limiter.limit("10 per minute")
def research_answer():
    """
    POST /api/ai/research/answer
    End-to-end grounded reasoning endpoint:
    Phase 1 (Search & Fetch) -> Phase 2 (Chunk & Rerank) -> Phase 3 (Reason & Grounded Citations).
    Authenticated: requires valid JWT Bearer access token.
    Rate-limited: 10 requests per IP per minute.
    Accepts: { "query": str, "max_results": Optional[int], "search_mode": Optional[str], "max_evidence_items": Optional[int], "max_evidence_chars": Optional[int] }
    Returns: AnswerResponse JSON with grounded answer text, verified citations, and generation metadata.
    """
    # 1. Feature flag guards
    if not current_app.config.get("AI_ENGINE_ENABLED", False):
        return jsonify({
            "error": "ai_engine_disabled",
            "message": "AI Engine web research is currently disabled.",
        }), 503

    if not current_app.config.get("AI_ENGINE_LLM_ENABLED", False):
        return jsonify({
            "error": "ai_engine_llm_disabled",
            "message": "AI Engine reasoning is currently disabled.",
        }), 503

    # 2. Enforce request body size limit
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
        answer_req = AnswerRequest(**data)
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

    # 5. Check document feature flag if search_mode uses documents
    if getattr(answer_req, "search_mode", "web") in ("document", "hybrid"):
        if not current_app.config.get("AI_ENGINE_DOCUMENTS_ENABLED", True):
            return jsonify({
                "error": "ai_engine_documents_disabled",
                "message": "Document retrieval is currently disabled.",
            }), 503

    # 6. Execute end-to-end research, evidence, and grounded reasoning pipeline
    try:
        search_req = SearchRequest(
            query=answer_req.query,
            max_results=answer_req.max_results,
            fetch_content=True,
        )
        answer_resp = _engine.research_and_answer(
            search_req,
            max_evidence_items=answer_req.max_evidence_items,
            max_evidence_chars=answer_req.max_evidence_chars,
            search_mode=getattr(answer_req, "search_mode", "web"),
            user_id=getattr(g, "user_id", None),
        )
        if hasattr(answer_resp, "model_dump"):
            payload = answer_resp.model_dump()
        else:
            payload = answer_resp.dict()

        return jsonify(payload), 200

    except Exception as e:
        logger.error(
            "[AI Engine] Unhandled answer reasoning error for query=%r: %s",
            getattr(answer_req, "query", "unknown"),
            e,
            exc_info=True,
        )
        return jsonify({
            "error": "reasoning_failed",
            "message": "An error occurred while generating grounded research answer.",
        }), 500


# ==============================================================================
# Phase 4 Document Ingestion & Knowledge Base Endpoints
# ==============================================================================

@ai_engine_bp.route("/documents/upload", methods=["POST"])
@token_required
@limiter.limit("5 per minute")
def upload_document():
    """
    POST /api/ai/documents/upload
    Uploads and indexes a private user document (.pdf, .txt, .md).
    Authenticated: requires valid JWT Bearer access token.
    Rate-limited: 5 requests per IP/user per minute.
    Accepts: multipart/form-data with 'file' and optional 'title'.
    """
    if not current_app.config.get("AI_ENGINE_ENABLED", False):
        return jsonify({
            "error": "ai_engine_disabled",
            "message": "AI Engine is currently disabled.",
        }), 503

    if not current_app.config.get("AI_ENGINE_DOCUMENTS_ENABLED", True):
        return jsonify({
            "error": "ai_engine_documents_disabled",
            "message": "Document ingestion and private knowledge base are currently disabled.",
        }), 503

    user_id = getattr(g, "user_id", None)
    if not user_id:
        return jsonify({"error": "unauthorized", "message": "Authentication required."}), 401

    if "file" not in request.files:
        return jsonify({
            "error": "validation_error",
            "message": "Missing required file field in multipart request.",
        }), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({
            "error": "validation_error",
            "message": "No file selected for upload.",
        }), 400

    max_size = current_app.config.get("AI_ENGINE_MAX_DOC_SIZE_BYTES", 5242880)
    file_bytes = file.read()
    if len(file_bytes) > max_size:
        return jsonify({
            "error": "request_too_large",
            "message": f"Uploaded file exceeds maximum limit of {max_size} bytes ({max_size // 1048576}MB).",
        }), 413

    title = request.form.get("title")

    try:
        doc = _engine.document_service.upload_document(
            user_id=user_id,
            file_bytes=file_bytes,
            filename=file.filename,
            title=title,
        )
        return jsonify({
            "document": doc.to_dict(),
            "message": "Document uploaded and indexed successfully.",
        }), 201

    except QuotaExceededError as e:
        return jsonify({
            "error": "quota_exceeded",
            "message": str(e),
        }), 400
    except (DocumentParserError, ValueError) as e:
        return jsonify({
            "error": "validation_error",
            "message": str(e),
        }), 400
    except Exception as e:
        logger.error("[AI Engine] Document upload failed for user=%s: %s", user_id, e, exc_info=True)
        return jsonify({
            "error": "upload_failed",
            "message": "An error occurred while uploading and parsing the document.",
        }), 500


@ai_engine_bp.route("/documents", methods=["GET"])
@token_required
@limiter.limit("30 per minute")
def list_documents():
    """
    GET /api/ai/documents
    Lists all indexed private documents for the authenticated user.
    """
    if not current_app.config.get("AI_ENGINE_ENABLED", False):
        return jsonify({
            "error": "ai_engine_disabled",
            "message": "AI Engine is currently disabled.",
        }), 503

    if not current_app.config.get("AI_ENGINE_DOCUMENTS_ENABLED", True):
        return jsonify({
            "error": "ai_engine_documents_disabled",
            "message": "Document ingestion is currently disabled.",
        }), 503

    user_id = getattr(g, "user_id", None)
    if not user_id:
        return jsonify({"error": "unauthorized", "message": "Authentication required."}), 401

    try:
        docs = _engine.document_service.list_documents(user_id=user_id)
        doc_list = [d.to_dict() for d in docs]
        total_chunks = sum(d.chunk_count for d in docs)
        return jsonify({
            "documents": doc_list,
            "total_documents": len(doc_list),
            "total_chunks": total_chunks,
        }), 200
    except Exception as e:
        logger.error("[AI Engine] Failed to list documents for user=%s: %s", user_id, e, exc_info=True)
        return jsonify({
            "error": "list_failed",
            "message": "An error occurred while listing documents.",
        }), 500


@ai_engine_bp.route("/documents/<document_id>", methods=["DELETE"])
@token_required
@limiter.limit("10 per minute")
def delete_document(document_id: str):
    """
    DELETE /api/ai/documents/<document_id>
    Deletes a user-owned document and all associated chunks.
    """
    if not current_app.config.get("AI_ENGINE_ENABLED", False):
        return jsonify({
            "error": "ai_engine_disabled",
            "message": "AI Engine is currently disabled.",
        }), 503

    if not current_app.config.get("AI_ENGINE_DOCUMENTS_ENABLED", True):
        return jsonify({
            "error": "ai_engine_documents_disabled",
            "message": "Document ingestion is currently disabled.",
        }), 503

    user_id = getattr(g, "user_id", None)
    if not user_id:
        return jsonify({"error": "unauthorized", "message": "Authentication required."}), 401

    try:
        success = _engine.document_service.delete_document(user_id=user_id, document_id=document_id)
        if not success:
            return jsonify({
                "error": "not_found",
                "message": "Document not found or access denied.",
            }), 404

        return jsonify({
            "message": "Document deleted successfully.",
            "document_id": document_id,
        }), 200
    except Exception as e:
        logger.error("[AI Engine] Failed to delete document=%s for user=%s: %s", document_id, user_id, e, exc_info=True)
        return jsonify({
            "error": "delete_failed",
            "message": "An error occurred while deleting the document.",
        }), 500


# ==============================================================================
# Phase 7 Task-Aware Execution Endpoint
# ==============================================================================

@ai_engine_bp.route("/execute", methods=["POST"])
@ai_engine_bp.route("/task/execute", methods=["POST"])
@token_required
@limiter.limit("30 per minute")
def execute_task():
    """
    POST /api/ai/execute
    Centralized Task-Aware Execution Endpoint.
    Routes tasks (resume, classification, extraction, chat, research, salary_analysis,
    company_review, vision, document_vision, text_generation) to local or cloud models.
    Authenticated: requires valid JWT Bearer access token.
    Rate-limited: 30 requests per IP per minute.
    """
    # 1. Feature flag guard
    if not current_app.config.get("AI_ENGINE_ENABLED", False):
        return jsonify({
            "error": "ai_engine_disabled",
            "message": "AI Engine is currently disabled.",
        }), 503

    # 2. Enforce request body size limit
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
            "message": "Request body must be a non-empty JSON object containing 'task' and 'prompt'.",
        }), 400

    task = data.get("task")
    prompt = data.get("prompt")
    if not task or not isinstance(task, str) or not task.strip():
        return jsonify({
            "error": "validation_error",
            "message": "Missing or invalid 'task' field.",
        }), 400

    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        return jsonify({
            "error": "validation_error",
            "message": "Missing or invalid 'prompt' field.",
        }), 400

    system_prompt = data.get("system_prompt")
    images = data.get("images")
    temperature = data.get("temperature")
    max_tokens = data.get("max_tokens")
    allow_cloud_fallback = data.get("allow_cloud_fallback")
    complexity = data.get("complexity")

    # Security: Disallow user-controlled endpoint URLs and credentials
    cleaned_kwargs = {k: v for k, v in data.items() if k not in (
        "task", "prompt", "system_prompt", "images", "temperature", "max_tokens", "allow_cloud_fallback",
        "complexity", "base_url", "url", "provider_url", "endpoint", "api_key", "token"
    )}

    try:
        result = _engine.execute_task(
            task=task.strip(),
            prompt=prompt.strip(),
            system_prompt=system_prompt.strip() if isinstance(system_prompt, str) and system_prompt.strip() else None,
            images=images if isinstance(images, list) else None,
            temperature=float(temperature) if temperature is not None else None,
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            allow_cloud_fallback=bool(allow_cloud_fallback) if allow_cloud_fallback is not None else None,
            complexity=str(complexity).strip().lower() if isinstance(complexity, str) and complexity.strip() else None,
            **cleaned_kwargs,
        )

        payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        from app.ai_engine.reasoning.task_executor import _scrub_credentials
        if isinstance(payload.get("error_message"), str):
            payload["error_message"] = _scrub_credentials(payload["error_message"])
        status_code = 200 if result.success else 503
        return jsonify(payload), status_code

    except Exception as e:
        logger.error("[AI Engine] Unhandled task execution error for task=%s: %s", task, e, exc_info=True)
        return jsonify({
            "error": "execution_failed",
            "message": "An error occurred while executing AI task.",
        }), 500
