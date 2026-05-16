"""
app/routes/resume.py
NyrVexa — Resume Suite (Phase 6.0)

Endpoints:
  POST   /api/resume/analyze          — ATS analyzer (JD-aware, backwards-compatible)
  GET    /api/resume/history          — list past analyses + built resumes
  GET    /api/resume/templates        — list all active templates
  GET    /api/resume/templates/<id>   — single template metadata
  POST   /api/resume/build            — AI resume builder → PDF (base64)
  POST   /api/resume/build-jd         — alias of /build (JD mode via job_description field)
  POST   /api/resume/enhance          — standalone bullet enhancer
  POST   /api/resume/compile          — raw LaTeX → PDF (live editor)
  GET    /api/resume/history/<id>     — single saved UserResume
  PATCH  /api/resume/draft/<id>       — save LaTeX draft from live editor
  DELETE /api/resume/history/<id>     — delete saved resume

/analyze backwards-compatibility:
  Without job_description → same 6-field JSON as before (original clients unaffected)
  With job_description    → 10-field JD-aware JSON

File key for analyze is still "file" — existing clients don't break.
"""
import os
import io
import re
import json
import uuid
import base64
import logging

import pdfplumber
import requests
from flask import Blueprint, request, jsonify, g

from app.middleware.auth_middleware import token_required
from app.extensions import db
from app.models.resume_analysis import ResumeAnalysis
from app.models.resume_template import ResumeTemplate
from app.models.user_resume import UserResume
from app.services.resume_builder_service import build_resume, enhance_bullets
from app.services.latex_compiler import compile_latex

resume_bp = Blueprint("resume", __name__, url_prefix="/api/resume")
logger    = logging.getLogger(__name__)

# ── Groq config ───────────────────────────────────────────────────────────────
GROQ_KEY   = os.environ.get("GROQ_RESUME_API_KEY") or os.environ.get("GROQ_API_KEY", "")
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Prompts ───────────────────────────────────────────────────────────────────
_GENERIC_PROMPT = """
You are an expert ATS (Applicant Tracking System) and career coach.
Analyze the resume text below and return ONLY a valid JSON object —
no markdown, no backticks, no explanation outside the JSON.

Return exactly this structure:
{
  "overall_score": <integer 0-100>,
  "ats_status": "<one of: Excellent / Good / Needs Work / Poor>",
  "skills_found": ["skill1", "skill2"],
  "strengths": ["strength1", "strength2", "strength3"],
  "improvements": ["improvement1", "improvement2", "improvement3"],
  "missing_keywords": ["keyword1", "keyword2"]
}

Scoring:
- 85-100: Strong, ATS-optimised, clear impact statements
- 70-84:  Good but missing some keywords or quantification
- 50-69:  Needs improvement in structure or keywords
- 0-49:   Major gaps in formatting, keywords, or content
"""

# Replace _JD_SYSTEM_PROMPT with this:

_JD_SYSTEM_PROMPT = """
You are a strict ATS (Applicant Tracking System) scanner. Score how well a resume matches a Job Description.

SCORING RUBRIC — follow exactly, no exceptions:

KEYWORD MATCH (40 pts max):
- Extract every required technical skill from the JD
- Award points ONLY for skills explicitly present in the resume
- Each missing required skill reduces this score proportionally
- "Python" in JD but resume only has "scripting" = 0 credit

ROLE ALIGNMENT (25 pts max):
- Resume's job titles/target role matches JD role exactly = 25 pts
- Adjacent role (e.g. fullstack vs frontend) = max 15 pts
- Completely different domain (e.g. ML engineer vs frontend dev) = max 5 pts

EXPERIENCE RELEVANCE (20 pts max):
- Past roles/projects use technologies listed in the JD = full points
- Generic experience with no JD-specific tech = low points

ATS FORMAT (15 pts max):
- Standard sections, no tables/columns/graphics = 15 pts
- Deduct 5 pts per formatting issue

FINAL SCORE = sum of all four. Do NOT be generous. Do NOT round up.
90+ means resume was written specifically for this exact JD.
Below 40 means candidate is not qualified for this role.
A Prompt Engineer resume against a Frontend Developer JD should score 20-35.

Return ONLY valid JSON. No markdown. No explanation.
"""

# Replace _JD_USER_TEMPLATE with this:

_JD_USER_TEMPLATE = """RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

Return this exact JSON structure with your computed values:
{{
  "overall_score": <integer, sum of four rubric sections>,
  "ats_status": "<Excellent if 85+, Good if 70-84, Needs Work if 50-69, Poor if below 50>",
  "jd_match_score": <same as overall_score>,
  "role_match": "<exact / adjacent / different>",
  "section_scores": {{
    "keyword_match": <0-40>,
    "role_alignment": <0-25>,
    "experience_relevance": <0-20>,
    "ats_format": <0-15>
  }},
  "matched_keywords": ["only JD keywords that appear verbatim in resume"],
  "missing_keywords": ["required JD keywords not found in resume"],
  "skills_found": ["all skills found in resume"],
  "strengths": ["3 specific strengths relative to this JD"],
  "improvements": ["3 specific things to add to pass this JD's ATS"],
  "ats_issues": ["formatting issues that break parsers"],
  "recommended_additions": ["exact phrases to add to improve score"]
}}"""

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _groq(system: str, user: str) -> str:
    resp = requests.post(
        GROQ_URL,
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": 0.1,
            "max_tokens":  1024,
        },
        headers={"Authorization": f"Bearer {GROQ_KEY}"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _parse_json(raw: str) -> dict:
    clean = re.sub(r"```json|```", "", raw).strip()
    return json.loads(clean)


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts)


def _ok(data: dict, code: int = 200):
    return jsonify({"success": True, **data}), code


def _err(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


# ─────────────────────────────────────────────────────────────────────────────
# 1. POST /api/resume/analyze
# ─────────────────────────────────────────────────────────────────────────────

@resume_bp.route("/analyze", methods=["POST"])
@token_required
def analyze_resume():
    user_id = g.user_id

    if "file" not in request.files:
        return _err("No file uploaded. Send multipart/form-data with key 'file'")

    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return _err("Only PDF files are supported")

    pdf_bytes = file.read()
    if len(pdf_bytes) > 5 * 1024 * 1024:
        return _err("File too large. Max 5MB")

    resume_text = _extract_pdf_text(pdf_bytes)
    if not resume_text.strip():
        return _err("Could not extract text from PDF. Make sure it is not a scanned image.")

    jd_text = request.form.get("job_description", "").strip()

    try:
        if jd_text:
            raw = _groq(
                _JD_SYSTEM_PROMPT,
                _JD_USER_TEMPLATE.format(resume_text=resume_text, jd_text=jd_text),
            )
        else:
            raw = _groq(_GENERIC_PROMPT, f"Resume text:\n\n{resume_text}")
        logger.info(f"[Analyze] Raw Groq response: {raw}")    
    except Exception as e:
        logger.error(f"[Analyze] Groq failed: {e}")
        return _err("AI analysis failed. Please try again.", 500)

    try:
        data = _parse_json(raw)
    except Exception as e:
        logger.error(f"[Analyze] JSON parse failed: {e}\nRaw: {raw[:300]}")
        return _err("Could not parse AI response. Try again.", 500)

    # Save to DB using existing ResumeAnalysis model — columns unchanged
    analysis = None
    try:
        analysis = ResumeAnalysis(
            user_id          = user_id,
            filename         = file.filename,
            score            = data.get("overall_score"),
            ats_status       = data.get("ats_status"),
            skills_found     = data.get("skills_found", []),
            strengths        = data.get("strengths", []),
            improvements     = data.get("improvements", []),
            missing_keywords = data.get("missing_keywords", []),
            raw_response     = raw,
        )
        db.session.add(analysis)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"[Analyze] DB save failed: {e}")

    # Base response — same fields as original
    response = {
        "analysis_id":      str(analysis.id) if analysis else None,
        "overall_score":    data.get("overall_score"),
        "ats_status":       data.get("ats_status"),
        "skills_found":     data.get("skills_found", []),
        "strengths":        data.get("strengths", []),
        "improvements":     data.get("improvements", []),
        "missing_keywords": data.get("missing_keywords", []),
        "model_used":       "groq/llama-3.3-70b-versatile",
        "jd_mode":          bool(jd_text),
    }
    # Extra fields only in JD mode
    if jd_text:
        response.update({
            "jd_match_score":       data.get("jd_match_score"),
            "matched_keywords":     data.get("matched_keywords", []),
            "ats_issues":           data.get("ats_issues", []),
            "recommended_additions": data.get("recommended_additions", []),
        })

    return jsonify(response), 200


# ─────────────────────────────────────────────────────────────────────────────
# 2. GET /api/resume/history
# ─────────────────────────────────────────────────────────────────────────────

@resume_bp.route("/history", methods=["GET"])
@token_required
def resume_history():
    analyses = (
        ResumeAnalysis.query
        .filter_by(user_id=g.user_id)
        .order_by(ResumeAnalysis.created_at.desc())
        .limit(10)
        .all()
    )
    return jsonify([{
        "id":         str(a.id),
        "filename":   a.filename,
        "score":      a.score,
        "ats_status": a.ats_status,
        "created_at": a.created_at.isoformat(),
    } for a in analyses]), 200


# ─────────────────────────────────────────────────────────────────────────────
# 3. GET /api/resume/templates
# ─────────────────────────────────────────────────────────────────────────────

@resume_bp.route("/templates", methods=["GET"])
def list_templates():
    templates = ResumeTemplate.query.filter_by(is_active=True).all()
    return _ok({"templates": [t.to_dict() for t in templates]})


# ─────────────────────────────────────────────────────────────────────────────
# 4. GET /api/resume/templates/<template_id>
# ─────────────────────────────────────────────────────────────────────────────

@resume_bp.route("/templates/<template_id>", methods=["GET"])
def get_template(template_id):
    template = ResumeTemplate.query.get(template_id)
    if not template or not template.is_active:
        return _err("Template not found", 404)
    return _ok({"template": template.to_dict(include_latex=False)})


# ─────────────────────────────────────────────────────────────────────────────
# 5 & 6. POST /api/resume/build  +  /api/resume/build-jd
# ─────────────────────────────────────────────────────────────────────────────

def _handle_build():
    """Shared logic for /build and /build-jd."""
    form_data = request.get_json(silent=True)
    if not form_data:
        return _err("Request body must be JSON")
    if not form_data.get("template_id"):
        return _err("template_id is required")
    if not form_data.get("full_name") or not form_data.get("email"):
        return _err("full_name and email are required")

    template = ResumeTemplate.query.get(form_data["template_id"])
    if not template or not template.is_active:
        return _err("Template not found or inactive", 404)

    resume_id = str(uuid.uuid4())
    user_id   = str(g.user_id)

    try:
        result = build_resume(
            form_data   = form_data,
            template_id = str(template.id),   # UUID string
            user_id     = user_id,
            resume_id   = resume_id,
        )
    except FileNotFoundError as e:
        logger.error(f"[Build] Tectonic not found: {e}")
        return _err("LaTeX compiler not installed on server.", 500)
    except Exception as e:
        logger.error(f"[Build] Failed: {e}", exc_info=True)
        return _err(f"Resume generation failed: {str(e)}", 502)

    # Save to user_resumes
    try:
        user_resume = UserResume(
            id          = resume_id,
            user_id     = g.user_id,
            template_id = template.id,
            resume_data = form_data,
            ai_content  = result["ai_content"],
            latex_draft = result["latex_code"],
            pdf_url     = result["pdf_url"],
            jd_text     = form_data.get("job_description"),
            status      = "compiled",
        )
        db.session.add(user_resume)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"[Build] DB save failed (non-fatal): {e}")

    return _ok({
        "resume_id":  resume_id,
        "pdf_base64": base64.b64encode(result["pdf_bytes"]).decode("utf-8"),
        "pdf_url":    result["pdf_url"],
        "ai_content": result["ai_content"],
        "latex_code": result["latex_code"],  
        "word_count": result["ai_content"].get("word_count", 0),
    })


@resume_bp.route("/build", methods=["POST"])
@token_required
def build_resume_endpoint():
    return _handle_build()


@resume_bp.route("/build-jd", methods=["POST"])
@token_required
def build_resume_jd():
    return _handle_build()


# ─────────────────────────────────────────────────────────────────────────────
# 7. POST /api/resume/enhance
# ─────────────────────────────────────────────────────────────────────────────

@resume_bp.route("/enhance", methods=["POST"])
@token_required
def enhance_bullets_endpoint():
    data = request.get_json(silent=True)
    if not data or "bullets" not in data:
        return _err("bullets array is required")
    bullets = data["bullets"]
    if not isinstance(bullets, list) or not bullets:
        return _err("bullets must be a non-empty array")
    try:
        enhanced = enhance_bullets(bullets)
    except Exception as e:
        logger.error(f"[Enhance] Failed: {e}")
        return _err(f"Enhancement failed: {str(e)}", 502)
    return _ok({"enhanced_bullets": enhanced})


# ─────────────────────────────────────────────────────────────────────────────
# 8. POST /api/resume/compile  — Live LaTeX Editor
# ─────────────────────────────────────────────────────────────────────────────

@resume_bp.route("/compile", methods=["POST"])
@token_required
def compile_latex_endpoint():
    data = request.get_json(silent=True)
    if not data or "latex_code" not in data:
        return _err("latex_code is required")
    if len(data["latex_code"]) < 50:
        return _err("latex_code is too short to be a valid LaTeX document")
    try:
        pdf_bytes = compile_latex(data["latex_code"])
    except FileNotFoundError as e:
        return _err(f"Tectonic not installed on server: {str(e)}", 500)
    except RuntimeError as e:
        return _err(f"LaTeX compilation failed: {str(e)}", 422)
    return _ok({"pdf_base64": base64.b64encode(pdf_bytes).decode("utf-8")})


# ─────────────────────────────────────────────────────────────────────────────
# 9. GET /api/resume/history/<resume_id>
# ─────────────────────────────────────────────────────────────────────────────

@resume_bp.route("/history/<resume_id>", methods=["GET"])
@token_required
def get_resume(resume_id):
    resume = UserResume.query.get(resume_id)
    if not resume:
        return _err("Resume not found", 404)
    if str(resume.user_id) != str(g.user_id):
        return _err("Access denied", 403)
    return _ok({"resume": resume.to_dict(include_ai_content=True, include_latex=True)})


# ─────────────────────────────────────────────────────────────────────────────
# 10. PATCH /api/resume/draft/<resume_id>
# ─────────────────────────────────────────────────────────────────────────────

@resume_bp.route("/draft/<resume_id>", methods=["PATCH"])
@token_required
def save_draft(resume_id):
    resume = UserResume.query.get(resume_id)
    if not resume:
        return _err("Resume not found", 404)
    if str(resume.user_id) != str(g.user_id):
        return _err("Access denied", 403)
    data = request.get_json(silent=True)
    if not data or "latex_code" not in data:
        return _err("latex_code is required")
    resume.latex_draft = data["latex_code"]
    resume.status      = "draft"
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return _err(f"Failed to save draft: {str(e)}", 500)
    return _ok({"message": "Draft saved", "resume_id": resume_id})


# ─────────────────────────────────────────────────────────────────────────────
# 11. DELETE /api/resume/history/<resume_id>
# ─────────────────────────────────────────────────────────────────────────────

@resume_bp.route("/history/<resume_id>", methods=["DELETE"])
@token_required
def delete_resume(resume_id):
    resume = UserResume.query.get(resume_id)
    if not resume:
        return _err("Resume not found", 404)
    if str(resume.user_id) != str(g.user_id):
        return _err("Access denied", 403)
    try:
        db.session.delete(resume)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return _err(f"Failed to delete: {str(e)}", 500)
    return _ok({"message": "Resume deleted", "resume_id": resume_id})

# ─────────────────────────────────────────────────────────────────────────────
# 12. POST /api/resume/regenerate/<resume_id>
# ─────────────────────────────────────────────────────────────────────────────
@resume_bp.route("/regenerate/<resume_id>", methods=["POST"])
@token_required
def regenerate_resume(resume_id):
    resume = UserResume.query.get(resume_id)
    if not resume:
        return _err("Resume not found", 404)
    if str(resume.user_id) != str(g.user_id):
        return _err("Access denied", 403)

    form_data = request.get_json(silent=True)
    if not form_data:
        return _err("Request body must be JSON")

    template = ResumeTemplate.query.get(str(resume.template_id))
    if not template or not template.is_active:
        return _err("Template not found", 404)

    try:
        result = build_resume(
            form_data   = form_data,
            template_id = str(resume.template_id),
            user_id     = str(g.user_id),
            resume_id   = resume_id,
        )
    except Exception as e:
        logger.error(f"[Regenerate] Failed: {e}", exc_info=True)
        return _err(f"Regeneration failed: {str(e)}", 502)

    try:
        resume.resume_data = form_data
        resume.ai_content  = result["ai_content"]
        resume.latex_draft = result["latex_code"]
        resume.pdf_url     = result["pdf_url"]
        resume.status      = "compiled"
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"[Regenerate] DB update failed: {e}")

    return _ok({
        "resume_id":  resume_id,
        "pdf_base64": base64.b64encode(result["pdf_bytes"]).decode("utf-8"),
        "pdf_url":    result["pdf_url"],
        "ai_content": result["ai_content"],
        "latex_code": result["latex_code"],
        "word_count": result["ai_content"].get("word_count", 0),
    })