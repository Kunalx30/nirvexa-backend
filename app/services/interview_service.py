"""
app/services/interview_service.py
NyrVexa — Phase 6.6 + 6B: Text Interview Prep + Voice Interview AI Service
"""
import logging
import json
import uuid
from datetime import datetime, timezone

import openai
from flask import current_app
from app.extensions import db
from app.models import InterviewSession, InterviewResponse

logger = logging.getLogger(__name__)


# ── Phase 6.6 — Text Interview Prep ──────────────────────────────────────────

def generate_questions(role: str, prep_type: str) -> dict:
    if prep_type == 'hr':
        type_instruction = (
            "Generate 10 HR/behavioral interview questions for this role. "
            "Focus on: teamwork, conflict resolution, strengths/weaknesses, "
            "career goals, situational judgment, and cultural fit. "
            "Tailor questions to Indian fresher and mid-level interview styles."
        )
    else:
        type_instruction = (
            "Generate 10 technical interview questions for this role. "
            "Focus on: core technical skills, problem-solving, tools/technologies, "
            "real-world scenarios, and role-specific knowledge. "
            "Mix easy, medium, and hard difficulty questions."
        )

    prompt = f"""You are an expert interview coach for the Indian job market.

Role: {role}
Interview Type: {prep_type.upper()}

{type_instruction}

Respond ONLY with valid JSON. No preamble, no markdown, no backticks.

{{
  "questions": [
    {{
      "id": 1,
      "question": "question text here",
      "difficulty": "easy | medium | hard",
      "what_interviewer_wants": "one sentence on what they are testing"
    }}
  ],
  "tips": [
    "one general tip for this type of interview",
    "another tip"
  ]
}}

Rules:
- Exactly 10 questions
- Questions must be specific to the {role} role
- difficulty must be one of: easy, medium, hard
- what_interviewer_wants must be concise — one sentence max"""

    try:
        client = _deepseek_client()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an interview coach. Always respond with valid JSON only. No markdown, no backticks."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7,
        )
        result = _parse_json(response)
        return {"success": True, "questions": result}

    except json.JSONDecodeError as e:
        logger.error("[InterviewService] JSON parse failed in generate_questions: %s", e)
        return {"success": False, "error": "AI returned malformed response. Please try again."}
    except Exception as e:
        logger.error("[InterviewService] generate_questions failed: %s", e)
        return {"success": False, "error": "Failed to generate questions. Please try again."}


def evaluate_answer(question: str, answer: str, role: str) -> dict:
    prompt = f"""You are an expert interview evaluator for the Indian job market.

Role being interviewed for: {role}
Interview Question: {question}
Candidate's Answer: {answer}

Evaluate this answer and respond ONLY with valid JSON. No preamble, no markdown, no backticks.

{{
  "score": 75,
  "grade": "B+",
  "feedback": "2-3 sentences of specific, constructive feedback on the answer",
  "strengths": ["what the candidate did well 1", "strength 2"],
  "improvements": ["specific improvement 1", "improvement 2"],
  "keywords_used": ["keyword found in answer 1", "keyword 2"],
  "keywords_missed": ["important keyword not in answer 1", "keyword 2"],
  "suggested_answer": "A model answer for this question in 3-5 sentences, tailored to the role",
  "verdict": "Strong | Acceptable | Needs Work"
}}

Rules:
- score must be a realistic integer 0-100
- grade: A+ (95+), A (90+), B+ (80+), B (70+), C (60+), D (below 60)
- feedback must be specific to THIS answer, not generic
- verdict: Strong if score >= 80, Acceptable if 60-79, Needs Work if below 60"""

    try:
        client = _deepseek_client()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an interview evaluator. Always respond with valid JSON only. No markdown, no backticks."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        result = _parse_json(response)
        return {"success": True, "data": result}

    except json.JSONDecodeError as e:
        logger.error("[InterviewService] JSON parse failed in evaluate_answer: %s", e)
        return {"success": False, "error": "AI returned malformed response. Please try again."}
    except Exception as e:
        logger.error("[InterviewService] evaluate_answer failed: %s", e)
        return {"success": False, "error": "Failed to evaluate answer. Please try again."}


# ── Phase 6B — Voice Interview AI ────────────────────────────────────────────

def generate_voice_questions(role: str, mode: str, difficulty: str) -> dict:
    """Generate questions for voice interview. Returns a flat list of question strings."""

    interviewer_style = (
        "Use the voice of a friendly Indian female HR interviewer named Ananya. "
        "Keep every question short, simple, natural, and easy to speak aloud. "
        "The first question must be: Tell me about yourself. "
    )

    mode_prompts = {
        'hr': (
            interviewer_style +
            f"Generate 12 HR/behavioral interview questions for a {role} role at {difficulty} difficulty. "
            "Focus on: teamwork, leadership, conflict resolution, strengths/weaknesses, career goals. "
            "Tailor to Indian fresher and mid-level interview styles. "
            "Return ONLY a JSON array of 12 question strings."
        ),
        'technical': (
            interviewer_style +
            f"Generate 12 technical interview questions for a {role} role at {difficulty} difficulty. "
            "Focus on: core technical concepts, problem-solving, tools, real-world scenarios. "
            "Return ONLY a JSON array of 12 question strings."
        ),
        'stress': (
            interviewer_style +
            f"Generate 18 rapid-fire short stress interview questions for a {role} role. "
            "Questions should be quick, direct, and slightly challenging. "
            "Return ONLY a JSON array of 18 question strings."
        ),
        'mock': (
            interviewer_style +
            f"Generate a mock interview for a {role} role at {difficulty} difficulty: "
            "first 6 HR/behavioral questions, then 6 technical questions. "
            "Return ONLY a JSON array of 12 question strings in order."
        ),
    }

    prompt = f"""{mode_prompts[mode]}

Respond ONLY with a valid JSON array of strings. No preamble, no markdown, no backticks. Example:
["Question 1?", "Question 2?", "Question 3?"]"""

    try:
        client = _deepseek_client()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an interview question generator. Always respond with a valid JSON array of strings only. No markdown, no backticks."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        questions = json.loads(raw)
        if not isinstance(questions, list):
            raise ValueError("Response is not a list")
        questions = _ensure_self_intro_question(questions)

        return {"success": True, "questions": questions}

    except Exception as e:
        logger.error("[InterviewService] generate_voice_questions failed: %s", e)
        return {"success": False, "error": "Failed to generate questions. Please try again."}


def evaluate_voice_answer(
    question: str,
    transcript: str,
    role: str,
    duration_seconds: int,
    word_count: int,
    filler_count: int,
) -> dict:
    """Full 6-metric voice answer evaluation."""

    wpm = round((word_count / duration_seconds) * 60) if duration_seconds > 0 else 0

    prompt = f"""You are an expert interview evaluator for the Indian job market.

Role: {role}
Question: {question}
Candidate's spoken answer (transcript): {transcript}

Additional metrics:
- Speaking duration: {duration_seconds} seconds
- Word count: {word_count}
- Words per minute: {wpm}
- Filler words detected: {filler_count} (um, uh, like, you know, basically, literally)

Evaluate this voice interview answer and respond ONLY with valid JSON. No preamble, no markdown, no backticks.

{{
  "content_score": 75,
  "keyword_score": 60,
  "grammar_score": 80,
  "confidence_score": 70,
  "pace_score": 85,
  "completeness_score": 65,
  "overall_score": 73,
  "grade": "B",
  "keywords_found": ["keyword 1", "keyword 2"],
  "keywords_missing": ["important keyword 1", "keyword 2"],
  "feedback": "2-3 sentences of specific, actionable feedback on the spoken answer",
  "suggested_answer": "A model spoken answer in 4-6 sentences, natural and role-specific",
  "strengths": ["strength 1", "strength 2"],
  "improvements": ["improvement 1", "improvement 2"],
  "pace_feedback": "comment on speaking pace — too fast, too slow, or good",
  "filler_feedback": "comment on filler word usage",
  "verdict": "Strong | Acceptable | Needs Work"
}}

Scoring rules:
- content_score (30% weight): Did the answer address the question correctly?
- keyword_score (20% weight): Were important domain keywords mentioned?
- grammar_score (15% weight): Were sentences well-structured?
- confidence_score (15% weight): Did the answer sound confident and clear?
- pace_score (10% weight): WPM of {wpm} — ideal is 130-160 WPM
- completeness_score (10% weight): Did the answer have opening, body, conclusion?
- overall_score: weighted average of all 6 scores
- grade: A+ (95+), A (90+), B+ (80+), B (70+), C (60+), D (below 60)
- verdict: Strong if overall >= 80, Acceptable if 60-79, Needs Work if below 60
- filler_feedback: {filler_count} fillers detected — comment appropriately"""

    try:
        client = _deepseek_client()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a voice interview evaluator. Always respond with valid JSON only. No markdown, no backticks."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0.3,
        )
        result = _normalize_voice_evaluation(
            _parse_json(response),
            wpm=wpm,
            filler_count=filler_count,
            transcript=transcript,
            duration_seconds=duration_seconds,
            word_count=word_count,
        )
        return {"success": True, "data": result}

    except json.JSONDecodeError as e:
        logger.error("[InterviewService] JSON parse failed in evaluate_voice_answer: %s", e)
        return {"success": False, "error": "AI returned malformed response. Please try again."}
    except Exception as e:
        logger.error("[InterviewService] evaluate_voice_answer failed: %s", e)
        return {"success": False, "error": "Failed to evaluate answer. Please try again."}


def create_interview_session(
    user_id: str, role: str, mode: str, difficulty: str, question_count: int
) -> dict:
    try:
        session = InterviewSession(
            id=str(uuid.uuid4()),
            user_id=user_id,
            role=role,
            mode=mode,
            difficulty=difficulty,
            total_questions=question_count,
            status='in_progress',
            created_at=datetime.now(timezone.utc),
)
        db.session.add(session)
        db.session.commit()
        return {"success": True, "session_id": session.id}

    except Exception as e:
        db.session.rollback()
        logger.error("[InterviewService] create_interview_session failed: %s", e)
        return {"success": False, "error": "Failed to create session."}


def finalize_interview_session(
    session_id: str, user_id: str,
    total_score: float, avg_wpm: float, filler_word_count: int,
    metric_scores: dict | None = None,
) -> dict:
    try:
        session = InterviewSession.query.filter_by(id=session_id, user_id=user_id).first()
        if not session:
            return {"success": False, "error": "Session not found."}

        session.total_score      = total_score
        session.avg_wpm          = avg_wpm
        session.filler_word_count = filler_word_count
        metric_scores = metric_scores or {}
        session.content_score = metric_scores.get("content_score", session.content_score)
        session.keyword_score = metric_scores.get("keyword_score", session.keyword_score)
        session.grammar_score = metric_scores.get("grammar_score", session.grammar_score)
        session.confidence_score = metric_scores.get("confidence_score", session.confidence_score)
        session.status           = 'completed'
        session.completed_at     = datetime.now(timezone.utc)
        db.session.commit()
        return {"success": True}

    except Exception as e:
        db.session.rollback()
        logger.error("[InterviewService] finalize_interview_session failed: %s", e)
        return {"success": False, "error": "Failed to finalize session."}


def get_user_sessions(user_id: str) -> dict:
    try:
        sessions = (
            InterviewSession.query
            .filter_by(user_id=user_id)
            .order_by(InterviewSession.created_at.desc())
            .all()
        )
        return {
            "success": True,
            "sessions": [_serialize_session(s) for s in sessions]
        }

    except Exception as e:
        logger.error("[InterviewService] get_user_sessions failed: %s", e)
        return {"success": False, "error": "Failed to fetch sessions."}


def get_session_report(session_id: str, user_id: str) -> dict:
    try:
        session = InterviewSession.query.filter_by(id=session_id, user_id=user_id).first()
        if not session:
            return {"success": False, "error": "Session not found."}

        responses = (
            InterviewResponse.query
            .filter_by(session_id=session_id)
            .order_by(InterviewResponse.question_number)
            .all()
        )

        data = _serialize_session(session)
        data['responses'] = [_serialize_response(r) for r in responses]
        return {"success": True, "session": data}

    except Exception as e:
        logger.error("[InterviewService] get_session_report failed: %s", e)
        return {"success": False, "error": "Failed to fetch session report."}


def delete_interview_session(session_id: str, user_id: str) -> dict:
    try:
        session = InterviewSession.query.filter_by(id=session_id, user_id=user_id).first()
        if not session:
            return {"success": False, "error": "Session not found."}

        InterviewResponse.query.filter_by(session_id=session_id).delete()
        db.session.delete(session)
        db.session.commit()
        return {"success": True}

    except Exception as e:
        db.session.rollback()
        logger.error("[InterviewService] delete_interview_session failed: %s", e)
        return {"success": False, "error": "Failed to delete session."}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _deepseek_client():
    return openai.OpenAI(
        api_key=current_app.config["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        timeout=60.0,
    )


def _parse_json(response) -> dict:
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _ensure_self_intro_question(questions: list) -> list:
    cleaned = [str(q).strip() for q in questions if str(q).strip()]
    intro = "Tell me about yourself."
    if not cleaned:
        return [intro]
    first = cleaned[0].lower()
    if "tell me about yourself" in first or "introduce yourself" in first:
        cleaned[0] = intro
        return cleaned
    return [intro] + cleaned


def _clamp_score(value, default=0) -> int:
    try:
        score = round(float(value))
    except (TypeError, ValueError):
        score = default
    return max(0, min(100, score))


def _pace_score_from_wpm(wpm: int) -> int:
    if wpm <= 0:
        return 0
    if 125 <= wpm <= 165:
        return 100
    distance = 125 - wpm if wpm < 125 else wpm - 165
    return max(25, 100 - round(distance * 1.7))


def _pace_label_from_wpm(wpm: int) -> str:
    if wpm <= 0:
        return "not measured"
    if wpm < 105:
        return "too slow"
    if wpm < 125:
        return "slightly slow"
    if wpm <= 165:
        return "ideal"
    if wpm <= 185:
        return "slightly fast"
    return "too fast"


def _filler_score(filler_count: int, word_count: int) -> int:
    if word_count <= 0:
        return 0
    filler_rate = filler_count / word_count
    if filler_count == 0:
        return 100
    if filler_rate <= 0.02:
        return 90
    if filler_rate <= 0.05:
        return 75
    if filler_rate <= 0.08:
        return 60
    return 40


def _completeness_score_from_text(transcript: str) -> int:
    words = len((transcript or "").split())
    if words >= 90:
        return 85
    if words >= 55:
        return 72
    if words >= 30:
        return 58
    return 40


def _normalize_voice_evaluation(
    result: dict,
    wpm: int,
    filler_count: int,
    transcript: str,
    duration_seconds: int,
    word_count: int,
) -> dict:
    """Ensure the frontend always receives all 6 voice metric scores."""
    result = result if isinstance(result, dict) else {}
    word_count = word_count or len((transcript or "").split())
    filler_score = _filler_score(filler_count, word_count)
    defaults = {
        "content_score": 60,
        "keyword_score": 55,
        "grammar_score": 65,
        "confidence_score": max(35, round((80 - (filler_count * 4) + filler_score) / 2)),
        "pace_score": _pace_score_from_wpm(wpm),
        "completeness_score": _completeness_score_from_text(transcript),
    }

    for key, default in defaults.items():
        result[key] = _clamp_score(result.get(key), default)

    weighted = (
        result["content_score"] * 0.30
        + result["keyword_score"] * 0.20
        + result["grammar_score"] * 0.15
        + result["confidence_score"] * 0.15
        + result["pace_score"] * 0.10
        + result["completeness_score"] * 0.10
    )
    result["overall_score"] = _clamp_score(result.get("overall_score"), round(weighted))

    overall = result["overall_score"]
    result["grade"] = result.get("grade") or (
        "A+" if overall >= 95 else
        "A" if overall >= 90 else
        "B+" if overall >= 80 else
        "B" if overall >= 70 else
        "C" if overall >= 60 else
        "D"
    )
    result["verdict"] = result.get("verdict") or (
        "Strong" if overall >= 80 else "Acceptable" if overall >= 60 else "Needs Work"
    )
    result["keywords_found"] = result.get("keywords_found") or []
    result["keywords_missing"] = result.get("keywords_missing") or []
    result["feedback"] = result.get("feedback") or "Your answer was captured and scored. Add one clear example and a short conclusion to make it stronger."
    result["suggested_answer"] = result.get("suggested_answer") or "A strong answer should briefly explain the situation, your action, and the result. Keep it specific to the role and end with what you learned."
    result["strengths"] = result.get("strengths") or []
    result["improvements"] = result.get("improvements") or []
    pace_label = _pace_label_from_wpm(wpm)
    result["pace_feedback"] = (
        f"Measured pace: {wpm} WPM, which is {pace_label}. "
        "Aim for 125-165 WPM in interview answers."
    )
    result["filler_feedback"] = (
        f"Detected {filler_count} filler word{'s' if filler_count != 1 else ''}. "
        f"Filler score: {filler_score}/100."
    )
    result["measured_metrics"] = {
        "word_count": word_count,
        "duration_seconds": duration_seconds,
        "wpm": wpm,
        "ideal_wpm_min": 125,
        "ideal_wpm_max": 165,
        "pace_label": pace_label,
        "filler_count": filler_count,
        "filler_rate_percent": round((filler_count / word_count) * 100, 1) if word_count else 0,
        "filler_score": filler_score,
    }
    return result


def _serialize_session(s) -> dict:
    return {
        "id":               s.id,
        "role":             s.role,
        "mode":             s.mode,
        "difficulty":       getattr(s, 'difficulty', None),
        "question_count":   s.total_questions,
        "total_score":      s.total_score,
        "content_score":    getattr(s, "content_score", None),
        "keyword_score":    getattr(s, "keyword_score", None),
        "grammar_score":    getattr(s, "grammar_score", None),
        "confidence_score": getattr(s, "confidence_score", None),
        "avg_wpm":          s.avg_wpm,
        "filler_word_count": s.filler_word_count,
        "status":           s.status,
        "created_at":       s.created_at.isoformat() if s.created_at else None,
        "completed_at":     s.completed_at.isoformat() if s.completed_at else None,
    }


def _serialize_response(r) -> dict:
    return {
        "id":              r.id,
        "question_number": r.question_number,
        "question_text":   r.question_text,
        "transcript":      r.transcript,
        "overall_score":   r.score,
        "feedback":        r.feedback,
        "suggested_answer": r.suggested_answer,
        "created_at":      r.created_at.isoformat() if r.created_at else None,
    }
