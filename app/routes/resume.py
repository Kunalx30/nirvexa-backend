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
from app.middleware.rate_limiter import premium_required, rate_limit
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
    clean = re.sub(r"```json|```", "", raw or "").strip()
    match = re.search(r"(\{[\s\S]*\})", clean)
    if match:
        clean = match.group(1)
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


def _clamp_score(value, default=0) -> int:
    try:
        score = round(float(value))
    except (TypeError, ValueError):
        score = default
    return max(0, min(100, score))


def _status_from_score(score: int) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Needs Work"
    return "Poor"


def _as_list(value) -> list:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase = (phrase or "").strip().lower()
    if not phrase:
        return False
    pattern = r"(?<![a-z0-9+#.])" + re.escape(phrase) + r"(?![a-z0-9+#.])"
    return re.search(pattern, text.lower()) is not None


def _extract_skills(text: str) -> list:
    known = [
        "python", "java", "javascript", "typescript", "react", "angular", "vue", "node.js", "node",
        "express", "django", "flask", "fastapi", "spring", "sql", "mysql", "postgresql", "mongodb",
        "redis", "aws", "azure", "gcp", "docker", "kubernetes", "git", "linux", "html", "css",
        "tailwind", "figma", "rest api", "graphql", "machine learning", "deep learning", "nlp",
        "pandas", "numpy", "tensorflow", "pytorch", "scikit-learn", "excel", "power bi", "tableau",
        "data analysis", "statistics", "devops", "ci/cd", "terraform", "jenkins", "c++", "c#",
        "php", "laravel", "ruby", "go", "rust", "swift", "kotlin", "android", "ios",
    ]
    found = [skill for skill in known if _contains_phrase(text, skill)]
    return sorted(set(found), key=lambda item: (len(item), item))


def _extract_jd_keywords(jd_text: str) -> list:
    skills = _extract_skills(jd_text)
    explicit = re.findall(r"(?:required|must have|skills?|technologies?|tools?)[:\-\s]+([^\n.;]+)", jd_text, flags=re.I)
    for group in explicit:
        for part in re.split(r",|/|\|| and ", group):
            part = part.strip(" -•\t").lower()
            if 2 <= len(part) <= 35 and not re.search(r"\b(years?|experience|knowledge|strong|good)\b", part):
                skills.append(part)
    return sorted(set(skills), key=lambda item: (len(item), item))


def _role_terms(text: str) -> set:
    roles = {
        "frontend": ["frontend", "front end", "react developer", "ui developer"],
        "backend": ["backend", "back end", "api developer"],
        "fullstack": ["fullstack", "full stack"],
        "data": ["data analyst", "data scientist", "analytics", "business analyst"],
        "ml": ["machine learning", "ml engineer", "ai engineer", "prompt engineer"],
        "devops": ["devops", "site reliability", "sre", "cloud engineer"],
        "mobile": ["android", "ios", "mobile developer", "flutter", "react native"],
        "design": ["ui/ux", "ux designer", "product designer"],
    }
    lowered = text.lower()
    return {role for role, terms in roles.items() if any(term in lowered for term in terms)}


def _generic_resume_score(resume_text: str) -> tuple[int, dict]:
    text = resume_text.lower()
    word_count = len(re.findall(r"\w+", text))
    sections = ["experience", "education", "skills", "projects"]
    section_hits = sum(1 for section in sections if section in text)
    has_contact = bool(re.search(r"[\w.+-]+@[\w.-]+\.\w+", resume_text)) and bool(re.search(r"\+?\d[\d\s().-]{7,}", resume_text))
    has_metrics = len(re.findall(r"\b\d+%|\b\d+x|\b\d+\+|\b\d{2,}\b", resume_text))
    skills = _extract_skills(resume_text)

    structure_score = min(30, section_hits * 7 + (2 if word_count >= 250 else 0))
    contact_score = 10 if has_contact else 4
    skill_score = min(25, len(skills) * 3)
    impact_score = min(25, has_metrics * 5)
    length_score = 10 if 350 <= word_count <= 900 else 6 if 220 <= word_count <= 1100 else 3
    score = _clamp_score(structure_score + contact_score + skill_score + impact_score + length_score)

    details = {
        "section_scores": {
            "structure": structure_score,
            "contact": contact_score,
            "skills": skill_score,
            "impact": impact_score,
            "length": length_score,
        },
        "skills_found": skills,
    }
    return score, details


def _jd_resume_score(resume_text: str, jd_text: str) -> tuple[int, dict]:
    required = _extract_jd_keywords(jd_text)
    matched = [kw for kw in required if _contains_phrase(resume_text, kw)]
    missing = [kw for kw in required if kw not in matched]
    keyword_score = round((len(matched) / len(required)) * 40) if required else 0

    resume_roles = _role_terms(resume_text)
    jd_roles = _role_terms(jd_text)
    if jd_roles and resume_roles & jd_roles:
        role_match = "exact"
        role_score = 25
    elif jd_roles and resume_roles:
        role_match = "adjacent"
        role_score = 12
    else:
        role_match = "different"
        role_score = 5

    experience_hits = sum(1 for kw in matched if re.search(r"(experience|project|built|developed|implemented|worked).*" + re.escape(kw), resume_text, flags=re.I | re.S))
    experience_score = min(20, round((experience_hits / max(len(required), 1)) * 20) + (5 if matched else 0))

    text = resume_text.lower()
    standard_sections = sum(1 for section in ("experience", "education", "skills", "projects") if section in text)
    ats_format_score = min(15, 3 + standard_sections * 3)
    if re.search(r"\btable\b|\bgraphic\b|\bimage\b", text):
        ats_format_score = max(0, ats_format_score - 5)

    total = _clamp_score(keyword_score + role_score + experience_score + ats_format_score)
    return total, {
        "jd_match_score": total,
        "role_match": role_match,
        "section_scores": {
            "keyword_match": keyword_score,
            "role_alignment": role_score,
            "experience_relevance": experience_score,
            "ats_format": ats_format_score,
        },
        "matched_keywords": matched,
        "missing_keywords": missing,
        "skills_found": _extract_skills(resume_text),
    }


def _normalize_analysis(data: dict, resume_text: str, jd_text: str) -> dict:
    data = data if isinstance(data, dict) else {}
    if jd_text:
        score, details = _jd_resume_score(resume_text, jd_text)
        missing = details["missing_keywords"]
        matched = details["matched_keywords"]
        data.update(details)
        data["overall_score"] = score
        data["ats_status"] = _status_from_score(score)
        data["strengths"] = _as_list(data.get("strengths"))[:3] or [
            f"Matched {len(matched)} required JD keyword(s).",
            "Resume text was readable by the parser.",
            "Existing content can be tailored toward the target role.",
        ]
        data["improvements"] = _as_list(data.get("improvements"))[:3] or [
            "Add missing required JD skills with honest project or work evidence.",
            "Align the headline and recent projects with the exact target role.",
            "Use quantified bullets that connect tools to outcomes.",
        ]
        data["ats_issues"] = _as_list(data.get("ats_issues")) or ([] if details["section_scores"]["ats_format"] >= 12 else ["Missing or unclear standard resume sections."])
        data["recommended_additions"] = _as_list(data.get("recommended_additions")) or [f"Add evidence for: {kw}" for kw in missing[:5]]
        return data

    score, details = _generic_resume_score(resume_text)
    data["overall_score"] = score
    data["ats_status"] = _status_from_score(score)
    data["skills_found"] = details["skills_found"] or _as_list(data.get("skills_found"))
    data["strengths"] = _as_list(data.get("strengths"))[:3] or ["Resume text is parseable", "Core sections are partially present", "Skills can be extracted from the document"]
    data["improvements"] = _as_list(data.get("improvements"))[:3] or ["Add stronger quantified achievements", "Include standard sections: Experience, Projects, Skills, Education", "Add role-specific keywords from the jobs you are targeting"]
    data["missing_keywords"] = _as_list(data.get("missing_keywords"))
    data["section_scores"] = details["section_scores"]
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 1. POST /api/resume/analyze
# ─────────────────────────────────────────────────────────────────────────────

@resume_bp.route("/analyze", methods=["POST"])
@token_required
@rate_limit("resume_analysis")
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
    data = _normalize_analysis(data, resume_text, jd_text)

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
        "section_scores":   data.get("section_scores", {}),
        "model_used":       "groq/llama-3.3-70b-versatile",
        "jd_mode":          bool(jd_text),
    }
    # Extra fields only in JD mode
    if jd_text:
        response.update({
            "jd_match_score":       data.get("jd_match_score"),
            "role_match":           data.get("role_match"),
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
@premium_required("resume_build")
def build_resume_endpoint():
    return _handle_build()


@resume_bp.route("/build-jd", methods=["POST"])
@token_required
@premium_required("resume_build")
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
@premium_required("resume_download")
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
@premium_required("resume_build")
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
