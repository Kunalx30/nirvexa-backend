"""
app/routes/resume.py
NirVexa — Phase 5.8: Resume Analyzer
POST /api/resume/analyze  — accepts PDF, runs Groq analysis, returns JSON
GET  /api/resume/history  — returns past analyses for logged-in user
"""
import os, json, logging, requests, io
from flask import Blueprint, request, jsonify, g
from app.middleware.auth_middleware import token_required
from app.extensions import db
from app.models.resume_analysis import ResumeAnalysis

resume_bp = Blueprint("resume", __name__, url_prefix="/api/resume")
logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """
You are an expert ATS (Applicant Tracking System) and career coach.
Analyze the resume text below and return ONLY a valid JSON object —
no markdown, no backticks, no explanation outside the JSON.

Return exactly this structure:
{
  "overall_score": <integer 0-100>,
  "ats_status": "<one of: Excellent / Good / Needs Work / Poor>",
  "skills_found": ["skill1", "skill2", ...],
  "strengths": ["strength1", "strength2", "strength3"],
  "improvements": ["improvement1", "improvement2", "improvement3"],
  "missing_keywords": ["keyword1", "keyword2", ...]
}

Scoring guide:
- 85-100: Strong resume, ATS-optimised, clear impact statements
- 70-84:  Good but missing some keywords or quantification
- 50-69:  Needs improvement in structure or keywords
- 0-49:   Major gaps in formatting, keywords, or content

Be specific. missing_keywords should be real tech/role keywords not in the resume.
"""


@resume_bp.route("/analyze", methods=["POST"])
@token_required
def analyze_resume():
    user_id = g.user_id

    # 1. Validate file
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send as multipart/form-data with key 'file'"}), 400

    file = request.files["file"]

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    # 2. Read PDF
    pdf_bytes = file.read()
    if len(pdf_bytes) > 5 * 1024 * 1024:
        return jsonify({"error": "File too large. Max 5MB"}), 400

    # 3. Extract text from PDF
    resume_text = _extract_text(pdf_bytes)
    if not resume_text.strip():
        return jsonify({"error": "Could not extract text from PDF. Make sure it is not a scanned image."}), 400

    # 4. Call Groq
    try:
        result = _call_groq(resume_text, ANALYSIS_PROMPT)
    except Exception as e:
        logger.error(f"[Resume] Groq call failed: {e}")
        return jsonify({"error": "AI analysis failed. Please try again.", "detail": str(e)}), 500

    # 5. Parse JSON from Groq response
    try:
        clean = result.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            clean = clean.lstrip("json").strip()
        data = json.loads(clean)
    except Exception as e:
        logger.error(f"[Resume] JSON parse failed: {e}\nRaw: {result[:300]}")
        return jsonify({"error": "Could not parse AI response. Try again."}), 500

    # 6. Save to DB
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
            raw_response     = result,
        )
        db.session.add(analysis)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"[Resume] DB save failed: {e}")

    return jsonify({
        "analysis_id":      str(analysis.id) if analysis and analysis.id else None,
        "overall_score":    data.get("overall_score"),
        "ats_status":       data.get("ats_status"),
        "skills_found":     data.get("skills_found", []),
        "strengths":        data.get("strengths", []),
        "improvements":     data.get("improvements", []),
        "missing_keywords": data.get("missing_keywords", []),
        "model_used":       "groq/llama-3.3-70b-versatile",
    }), 200


@resume_bp.route("/history", methods=["GET"])
@token_required
def resume_history():
    user_id = g.user_id
    analyses = ResumeAnalysis.query.filter_by(user_id=user_id)\
                  .order_by(ResumeAnalysis.created_at.desc())\
                  .limit(10).all()

    return jsonify([{
        "id":         str(a.id),
        "filename":   a.filename,
        "score":      a.score,
        "ats_status": a.ats_status,
        "created_at": a.created_at.isoformat(),
    } for a in analyses]), 200


def _extract_text(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes using pdfplumber."""
    import pdfplumber
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)


def _call_groq(resume_text: str, prompt: str) -> str:
    """Send resume text to Groq Llama 3.3 70B, return raw response."""
    GROQ_RESUME_API_KEY = os.environ.get("GROQ_RESUME_API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user",   "content": f"Resume text:\n\n{resume_text}"}
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    resp = requests.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {GROQ_RESUME_API_KEY}"},
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]