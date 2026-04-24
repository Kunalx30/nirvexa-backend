"""
app/routes/resume.py
NirVexa — Phase 5.8: Resume Analyzer
POST /api/resume/analyze  — accepts PDF, runs Gemini analysis, returns JSON
GET  /api/resume/history  — returns past analyses for logged-in user
"""
import os, base64, json, logging, requests
from flask import Blueprint, request, jsonify, g
from app.middleware.auth_middleware import token_required
from app.extensions import db
from app.models.resume_analysis import ResumeAnalysis

resume_bp = Blueprint("resume", __name__, url_prefix="/api/resume")
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

ANALYSIS_PROMPT = """
You are an expert ATS (Applicant Tracking System) and career coach.
Analyze the resume in the attached PDF and return ONLY a valid JSON object — 
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

    # 2. Read + base64 encode
    pdf_bytes = file.read()
    if len(pdf_bytes) > 5 * 1024 * 1024:
        return jsonify({"error": "File too large. Max 5MB"}), 400

    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

    # 3. Call Gemini
    try:
        result = _call_gemini(pdf_b64, ANALYSIS_PROMPT)
    except Exception as e:
        logger.error(f"[Resume] Gemini call failed: {e}")
        return jsonify({"error": "AI analysis failed. Please try again."}), 500

    # 4. Parse JSON from Gemini response
    try:
        clean = result.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[-2] if "```" in clean[3:] else clean
            clean = clean.lstrip("json").strip()
        data = json.loads(clean)
    except Exception as e:
        logger.error(f"[Resume] JSON parse failed: {e}\nRaw: {result[:300]}")
        return jsonify({"error": "Could not parse AI response. Try again."}), 500

    # 5. Save to DB
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
        "model_used":       "gemini-1.5-flash",
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


def _call_gemini(pdf_b64: str, prompt: str) -> str:
    """Send PDF + prompt to Gemini 1.5 Flash, return raw text response."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"
    )

    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "application/pdf", "data": pdf_b64}},
                {"text": prompt}
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 1024,
        }
    }

    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]