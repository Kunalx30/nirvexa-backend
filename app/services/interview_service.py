"""
app/services/interview_service.py
NyrVexa — Phase 6.6 + 6B: Text Interview Prep + Voice Interview AI Service
"""
import logging
import json
import uuid
import re
from datetime import datetime, timezone

import google.generativeai as genai
import groq as groq_sdk
import openai
from flask import current_app
from mistralai import Mistral
from app.extensions import db
from app.models import InterviewSession, InterviewResponse

logger = logging.getLogger(__name__)


INTERVIEWER_PERSONA = """You are Anya, a professional senior HR & technical interviewer at a top Indian tech company.

Your personality:
- Warm but professional — like a real Indian woman interviewer
- Encouraging after good answers, gently probing after weak ones
- Natural conversational style — not robotic, not scripted
- Uses natural Indian English patterns occasionally, such as "so", "right", "absolutely", and "that's good"
- Asks follow-up probes naturally
- NEVER repeats the exact same reaction twice

Your role: conduct a realistic job interview that feels like a real human conversation.
The candidate wants a real interview experience, not a simulation."""


# ── Phase 6.6 — Text Interview Prep ──────────────────────────────────────────

def generate_questions(role: str, prep_type: str) -> dict:
    role = (role or "").strip()
    if not role:
        return {"success": False, "error": "Job role is required."}

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
        result = _call_llm_json(
            prompt,
            "You are an interview coach. Always respond with valid JSON only. No markdown, no backticks.",
            max_tokens=1500,
            temperature=0.7,
            prefer_deepseek=True,
        )
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
        result = _call_llm_json(
            prompt,
            "You are an interview evaluator. Always respond with valid JSON only. No markdown, no backticks.",
            max_tokens=1024,
            temperature=0.3,
            prefer_deepseek=True,
        )
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
    role = (role or "").strip()
    if not role:
        return {"success": False, "error": "Job role is required."}


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
        questions = _call_llm_json(
            prompt,
            "You are an interview question generator. Always respond with a valid JSON array of strings only. No markdown, no backticks.",
            max_tokens=1500,
            temperature=0.7,
        )
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
        result = _normalize_voice_evaluation(
            _call_llm_json(
                prompt,
                "You are a voice interview evaluator. Always respond with valid JSON only. No markdown, no backticks.",
                max_tokens=1200,
                temperature=0.3,
                prefer_deepseek=True,
            ),
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


def generate_interview_plan(job_role: str, experience_level: str, user_name: str) -> dict:
    job_role = (job_role or "").strip()
    if not job_role:
        return {"success": False, "error": "Job role is required."}
    user_name = _first_name_only(user_name)

    prompt = f"""You are interviewing {user_name or 'the candidate'} for a {job_role} role ({experience_level} level).

Generate a natural interview opening greeting AND a full question plan.
For the greeting, use the candidate's first name at most once. Do not use a surname. Do not say hi/hello more than once.

Return ONLY valid JSON (no markdown, no explanation):
{{
  "greeting": "<2-3 sentence warm, natural greeting introducing yourself as Anya, mention the role, make the candidate feel at ease — vary the wording each time, don't be robotic>",
  "questions": [
    {{
      "id": 1,
      "text": "<the interview question — phrased naturally as a human interviewer would ask>",
      "type": "hr|technical|situational|cultural|closing",
      "category": "<specific sub-topic like 'introduction', 'python', 'data structures', 'conflict resolution', etc>",
      "follow_up_hint": "<what Anya should probe if answer is weak or vague>"
    }}
  ]
}}

Question mix for {job_role} ({experience_level}):
- 1 introduction question
- 4-5 technical core questions specific to {job_role}
- 2-3 technical depth questions
- 3 HR/behavioral questions
- 2 situational questions
- 1 cultural fit / motivation question
- 1 salary / notice period / availability question
- 1 closing (do you have questions for us)

Total: 15-17 questions. Make them SPECIFIC to {job_role}, not generic. Each question should sound exactly how a human interviewer would say it out loud."""

    try:
        data = _call_llm_json(prompt, INTERVIEWER_PERSONA, max_tokens=3000, temperature=0.85)
        if not isinstance(data, dict) or "greeting" not in data or "questions" not in data:
            raise ValueError("Invalid interview plan structure")
        questions = data.get("questions") or []
        data["questions"] = [
            {
                "id": idx + 1,
                "text": str(q.get("text") if isinstance(q, dict) else q).strip(),
                "type": (q.get("type") if isinstance(q, dict) else "hr") or "hr",
                "category": (q.get("category") if isinstance(q, dict) else f"question {idx + 1}") or f"question {idx + 1}",
                "follow_up_hint": (q.get("follow_up_hint") if isinstance(q, dict) else "") or "",
            }
            for idx, q in enumerate(questions)
            if str(q.get("text") if isinstance(q, dict) else q).strip()
        ]
        data["questions"] = _ensure_plan_intro_question(data["questions"])
        data["questions"] = _ensure_minimum_plan_questions(data["questions"], job_role, min_count=12)
        if len(data["questions"]) > 17:
            data["questions"] = data["questions"][:17]
            for idx, question in enumerate(data["questions"]):
                question["id"] = idx + 1
        data["greeting"] = _ensure_named_greeting(data.get("greeting", ""), user_name, job_role)
        data["total"] = len(data["questions"])
        data["job_role"] = job_role
        data["experience_level"] = experience_level
        return {"success": True, "data": data}
    except Exception as e:
        logger.error("[InterviewService] generate_interview_plan failed: %s", e)
        return {"success": False, "error": "Failed to start interview. Please try again."}


def generate_reaction_and_question(
    job_role: str,
    question: str,
    user_answer: str,
    question_index: int,
    total_questions: int,
    all_qa: list,
) -> dict:
    is_last = question_index >= total_questions - 1
    recent = (all_qa or [])[-3:]
    context_str = "\n".join([f"Q: {qa.get('question', '')}\nA: {qa.get('answer', '')}" for qa in recent])

    if is_last:
        prompt = f"""Job role: {job_role}
Last question asked: "{question}"
Candidate's answer: "{user_answer}"

This was the FINAL question. Generate a warm, natural closing reaction from Anya (2-3 sentences max).
Tell the candidate the interview is complete and you'll be sharing feedback shortly.

Return ONLY valid JSON:
{{
  "reaction": "<warm closing reaction — thank the candidate, say you'll compile feedback>",
  "next_question": null,
  "is_last": true
}}"""
    else:
        prompt = f"""Job role: {job_role}
Recent conversation:
{context_str}

Current question asked: "{question}"
Candidate's answer: "{user_answer}"
This is question {question_index + 1} of {total_questions}.

As Anya, generate:
1. A brief natural reaction to this specific answer (1-2 sentences ONLY — acknowledge what they said, good or needs improvement, BE SPECIFIC to their answer, never generic). The reaction MUST NEVER ask the candidate any clarifying questions, follow-up questions, or request elaboration (e.g., do NOT say "Can you explain...?", "Could you tell me more...?", or "How did you...?"), as the flow will immediately transition to the next question.
2. A smooth natural transition to the next question (1 sentence connector like "Moving on..." or "That's a good point, now let me ask you...")

Vary your reactions — don't always say "Great answer!" Use: "Interesting...", "That makes sense...", "I see, so you...", "Right, and...", "That's a solid approach...", "Good, I like that you mentioned...", etc.

Return ONLY valid JSON:
{{
  "reaction": "<specific reaction to their answer>",
  "transition": "<natural bridge to next question>",
  "next_question": null,
  "is_last": false
}}

Note: next_question will be injected from the pre-planned list — return null for it."""

    try:
        data = _call_llm_json(prompt, INTERVIEWER_PERSONA, max_tokens=500, temperature=0.9)
        data["is_last"] = is_last
        return {"success": True, "data": data}
    except Exception as e:
        logger.error("[InterviewService] generate_reaction_and_question failed: %s", e)
        return {"success": False, "error": "Failed to process answer. Please try again."}


def generate_full_interview_evaluation(job_role: str, experience_level: str, all_qa: list) -> dict:
    all_qa = _clean_all_qa(all_qa)
    qa_text = "\n\n".join([
        f"Q{i + 1}: {qa.get('question', '')}\nAnswer: {qa.get('answer', '[No answer given]')}"
        for i, qa in enumerate(all_qa)
    ])

    prompt = f"""You interviewed a candidate for {job_role} ({experience_level} level).

Here is the complete interview transcript:
{qa_text}

Provide a comprehensive evaluation. Return ONLY valid JSON:
{{
  "overall_score": <0-100 integer>,
  "hire_recommendation": "Strong Yes | Yes | Maybe | No",
  "summary": "<3-4 sentence honest overall summary specific to this interview>",
  "scores": {{
    "technical_knowledge": <0-100>,
    "communication_clarity": <0-100>,
    "confidence": <0-100>,
    "answer_relevance": <0-100>,
    "problem_solving": <0-100>,
    "cultural_fit": <0-100>
  }},
  "strengths": ["<specific strength with example>", "<specific strength>", "<specific strength>"],
  "improvements": [
    {{
      "area": "<specific area>",
      "issue": "<what exactly was weak or missing>",
      "how_to_fix": "<concrete actionable advice>",
      "resource": "<specific course/book/practice method>"
    }}
  ],
  "per_question": [
    {{
      "question_number": 1,
      "question": "<question text>",
      "score": <0-10>,
      "feedback": "<specific feedback>",
      "what_was_good": "<if anything>",
      "what_was_missing": "<if anything>"
    }}
  ],
  "next_steps": ["<priority action 1>", "<priority action 2>", "<priority action 3>"],
  "motivational_note": "<encouraging closing note from Anya>"
}}"""

    try:
        data = _call_llm_json(
            prompt,
            "You are Anya, an expert interviewer and career coach. Provide honest, detailed, actionable evaluation. Be specific — no generic advice. Your evaluation should feel like genuine professional feedback, not automated scoring.",
            max_tokens=4000,
            temperature=0.7,
            prefer_deepseek=True,
        )
        return {"success": True, "data": _normalize_full_interview_evaluation(data, job_role, experience_level, all_qa)}
    except Exception as e:
        logger.error("[InterviewService] generate_full_interview_evaluation failed: %s", e)
        fallback = _normalize_full_interview_evaluation({}, job_role, experience_level, all_qa)
        fallback["evaluation_source"] = "deterministic_fallback"
        return {"success": True, "data": fallback}


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

def _clean_all_qa(all_qa: list) -> list[dict]:
    cleaned = []
    for idx, qa in enumerate(all_qa or []):
        if not isinstance(qa, dict):
            continue
        question = str(qa.get("question") or qa.get("question_text") or f"Question {idx + 1}").strip()
        answer = str(qa.get("answer") or qa.get("user_answer") or qa.get("transcript") or "").strip()
        if question or answer:
            cleaned.append({"question": question, "answer": answer})
    return cleaned


def _answer_quality_score(question: str, answer: str) -> int:
    answer = (answer or "").strip()
    words = re.findall(r"[A-Za-z][A-Za-z0-9+#.-]*", answer.lower())
    if not words:
        return 0

    word_count = len(words)
    unique_ratio = len(set(words)) / max(word_count, 1)
    score = 25
    score += min(25, round(word_count * 0.45))
    score += 15 if any(ch.isdigit() for ch in answer) else 0
    score += 10 if any(token in answer.lower() for token in ("because", "therefore", "result", "impact", "improved", "reduced", "built", "designed")) else 0
    score += 10 if word_count >= 45 and "." in answer else 0
    if word_count < 12:
        score -= 20
    if unique_ratio < 0.45:
        score -= 10
    return _clamp_score(score)


def _recommendation_from_score(score: int) -> str:
    if score >= 85:
        return "Strong Yes"
    if score >= 72:
        return "Yes"
    if score >= 55:
        return "Maybe"
    return "No"


def _normalize_full_interview_evaluation(data: dict, job_role: str, experience_level: str, all_qa: list) -> dict:
    data = data if isinstance(data, dict) else {}
    per_question = []
    question_scores = []
    ai_per_question = data.get("per_question") if isinstance(data.get("per_question"), list) else []

    for idx, qa in enumerate(all_qa or []):
        deterministic_score = round(_answer_quality_score(qa.get("question", ""), qa.get("answer", "")) / 10)
        ai_item = ai_per_question[idx] if idx < len(ai_per_question) and isinstance(ai_per_question[idx], dict) else {}
        try:
            score_10 = round(float(ai_item.get("score", deterministic_score) or deterministic_score))
        except (TypeError, ValueError):
            score_10 = deterministic_score
        score_10 = max(0, min(10, score_10))
        if not qa.get("answer"):
            score_10 = 0
        question_scores.append(score_10 * 10)
        per_question.append({
            "question_number": idx + 1,
            "question": ai_item.get("question") or qa.get("question") or f"Question {idx + 1}",
            "score": score_10,
            "feedback": ai_item.get("feedback") or ("Answer was too short to judge." if not qa.get("answer") else "The answer was scored for relevance, detail, structure, and evidence."),
            "what_was_good": ai_item.get("what_was_good") or ("Clear attempt to answer the question." if score_10 >= 5 else ""),
            "what_was_missing": ai_item.get("what_was_missing") or ("Add a specific example, measurable result, and clearer role-specific details." if score_10 < 8 else "Minor polish in concision and evidence."),
        })

    if not question_scores:
        question_scores = [0]

    avg = round(sum(question_scores) / len(question_scores))
    scores = data.get("scores") if isinstance(data.get("scores"), dict) else {}
    normalized_scores = {
        "technical_knowledge": _clamp_score(scores.get("technical_knowledge"), avg),
        "communication_clarity": _clamp_score(scores.get("communication_clarity"), avg),
        "confidence": _clamp_score(scores.get("confidence"), max(0, avg - 5)),
        "answer_relevance": _clamp_score(scores.get("answer_relevance"), avg),
        "problem_solving": _clamp_score(scores.get("problem_solving"), max(0, avg - 8)),
        "cultural_fit": _clamp_score(scores.get("cultural_fit"), avg),
    }
    overall = _clamp_score(data.get("overall_score"), round(sum(normalized_scores.values()) / len(normalized_scores)))

    return {
        "success": True,
        "job_role": job_role,
        "experience_level": experience_level,
        "overall_score": overall,
        "hire_recommendation": data.get("hire_recommendation") or _recommendation_from_score(overall),
        "summary": data.get("summary") or f"This {job_role} interview was scored from {len(all_qa or [])} answered question(s). The result reflects answer relevance, specificity, communication, and evidence. Stronger answers should include concrete examples, role-specific terms, and measurable outcomes.",
        "scores": normalized_scores,
        "strengths": data.get("strengths") or ["Completed the interview flow", "Attempted to address the questions directly", "Provided enough signal for a structured evaluation"],
        "improvements": data.get("improvements") or [{
            "area": "Answer depth",
            "issue": "Several answers need more concrete evidence and role-specific detail.",
            "how_to_fix": "Use a short situation-action-result structure and mention tools, decisions, and measurable outcomes.",
            "resource": "Practice 5 STAR answers for the target role and review core role fundamentals.",
        }],
        "per_question": per_question,
        "next_steps": data.get("next_steps") or ["Rewrite the weakest two answers with examples", "Add measurable outcomes to project explanations", "Practice speaking each answer in 60-90 seconds"],
        "motivational_note": data.get("motivational_note") or "You have a clear baseline now. Tighten the examples and make every answer prove one job-relevant skill.",
        "evaluation_source": data.get("evaluation_source") or "ai_normalized",
    }

def _config_value(key: str) -> str:
    value = current_app.config.get(key) or ""
    return str(value).strip()


def _unique_keys(*keys: str) -> list[str]:
    seen = set()
    unique = []
    for key in keys:
        key = (key or "").strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def _llm_providers(prefer_deepseek: bool = False) -> list[dict]:
    groq_keys = _unique_keys(_config_value("GROQ_API_KEY"), _config_value("GROQ_INTERVIEW_API_KEY"))
    gemini_keys = _unique_keys(_config_value("GEMINI_API_KEY"), _config_value("GEMINI_INTERVIEW_API_KEY"))
    mistral_keys = _unique_keys(_config_value("MISTRAL_API_KEY"), _config_value("MISTRAL_INTERVIEW_API_KEY"))
    deepseek_key = _config_value("DEEPSEEK_API_KEY")

    providers = []
    if prefer_deepseek and deepseek_key:
        providers.append({"name": "DeepSeek", "type": "deepseek", "key": deepseek_key})

    providers.extend({"name": f"Groq #{idx + 1}", "type": "groq", "key": key} for idx, key in enumerate(groq_keys))

    if not prefer_deepseek and deepseek_key:
        providers.append({"name": "DeepSeek", "type": "deepseek", "key": deepseek_key})

    providers.extend({"name": f"Gemini #{idx + 1}", "type": "gemini", "key": key} for idx, key in enumerate(gemini_keys))
    providers.extend({"name": f"Mistral #{idx + 1}", "type": "mistral", "key": key} for idx, key in enumerate(mistral_keys))
    return providers


def _call_provider(provider: dict, prompt: str, system: str, max_tokens: int, temperature: float) -> str:
    provider_type = provider["type"]
    api_key = provider["key"]

    if provider_type == "groq":
        client = groq_sdk.Groq(api_key=api_key, timeout=60.0)
        resp = client.chat.completions.create(
            model=current_app.config.get("GROQ_MODEL_FAST", "llama-3.3-70b-versatile"),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    if provider_type == "deepseek":
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=90.0)
        resp = client.chat.completions.create(
            model=current_app.config.get("DEEPSEEK_MODEL", "deepseek-chat"),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    if provider_type == "gemini":
        genai.configure(api_key=api_key)
        from google.generativeai.types import HarmBlockThreshold, HarmCategory

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        model = genai.GenerativeModel(
            model_name=current_app.config.get("GEMINI_MODEL", "gemini-2.5-flash"),
            system_instruction=system,
            generation_config=genai.GenerationConfig(max_output_tokens=max_tokens, temperature=temperature),
        )
        resp = model.generate_content(prompt, safety_settings=safety_settings)
        return resp.text

    if provider_type == "mistral":
        client = Mistral(api_key=api_key, timeout_ms=60000)
        resp = client.chat.complete(
            model=current_app.config.get("MISTRAL_MODEL", "mistral-small-latest"),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    raise ValueError(f"Unknown provider type: {provider_type}")


def _call_llm(prompt: str, system: str, max_tokens: int = 2048, temperature: float = 0.7, prefer_deepseek: bool = False) -> str:
    errors = []
    providers = _llm_providers(prefer_deepseek=prefer_deepseek)
    if not providers:
        raise RuntimeError("No LLM providers configured. Please configure at least one of: GROQ_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY, or MISTRAL_API_KEY.")
    for provider in providers:

        try:
            logger.info("[InterviewService] Trying %s", provider["name"])
            return _call_provider(provider, prompt, system, max_tokens, temperature)
        except Exception as exc:
            logger.warning("[InterviewService] %s failed: %s", provider["name"], exc)
            errors.append(f"{provider['name']}: {exc}")
    raise RuntimeError("All interview LLM providers failed")


def _call_llm_json(prompt: str, system: str, max_tokens: int = 2048, temperature: float = 0.7, prefer_deepseek: bool = False):
    raw = _call_llm(prompt, system, max_tokens=max_tokens, temperature=temperature, prefer_deepseek=prefer_deepseek)
    return _parse_json(raw)


def _deepseek_client():
    return openai.OpenAI(
        api_key=current_app.config["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        timeout=60.0,
    )


def _parse_json(response):
    if isinstance(response, str):
        raw = response.strip()
    else:
        raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
    if match:
        raw = match.group(1)
    return json.loads(raw)


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


def _ensure_plan_intro_question(questions: list) -> list:
    intro = {
        "id": 1,
        "text": "Tell me about yourself.",
        "type": "hr",
        "category": "introduction",
        "follow_up_hint": "Ask for a concise summary of background, strengths, and role fit.",
    }
    if not questions:
        return [intro]
    first = questions[0].get("text", "").lower()
    if "tell me about yourself" in first or "introduce yourself" in first:
        questions[0] = {**questions[0], "id": 1, "text": "Tell me about yourself."}
    else:
        questions = [intro] + questions
    for idx, question in enumerate(questions):
        question["id"] = idx + 1
    return questions


def _ensure_named_greeting(greeting: str, user_name: str, job_role: str) -> str:
    greeting = (greeting or "").strip()
    name = _first_name_only(user_name)
    if not name or name.lower() == "there":
        return _clean_greeting(greeting) or f"Hi there, I am Anya. I will be taking your mock interview for the {job_role} role today. Take a breath and answer naturally."

    greeting = _clean_greeting(greeting, name)
    if greeting:
        return greeting

    return f"Hi {name}, I am Anya. I will be taking your mock interview for the {job_role} role today. Take a breath and answer naturally."


def _first_name_only(value: str) -> str:
    name = re.sub(r"\s+", " ", str(value or "")).strip()
    if not name or name.lower() == "there":
        return name
    return re.split(r"[\s@._-]+", name, maxsplit=1)[0].strip(" ,.!?") or name


def _clean_greeting(greeting: str, first_name: str = "") -> str:
    text = re.sub(r"\s+", " ", str(greeting or "")).strip()
    if not text:
        return ""

    if first_name:
        escaped = re.escape(first_name)
        text = re.sub(
            rf"^(hi|hello)\s+{escaped}(?:\s+[A-Za-z][A-Za-z'.-]*)*\s*,?\s*(hi|hello)\s+{escaped}\s*,?\s*",
            f"Hi {first_name}, ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            rf"^(hi|hello)\s+{escaped}\s+[A-Za-z][A-Za-z'.-]*(?:\s+[A-Za-z][A-Za-z'.-]*)*\s*,?",
            f"Hi {first_name},",
            text,
            flags=re.IGNORECASE,
        )
        if not re.search(rf"\b{escaped}\b", text, flags=re.IGNORECASE):
            text = re.sub(r"^(hi|hello)\s+(there|candidate)\s*,?\s*", "", text, flags=re.IGNORECASE)
            text = f"Hi {first_name}, {text[:1].lower()}{text[1:]}" if text else f"Hi {first_name},"

        seen = False

        def keep_first(match):
            nonlocal seen
            if not seen:
                seen = True
                return match.group(0)
            return ""

        text = re.sub(rf"\b{escaped}\b", keep_first, text, flags=re.IGNORECASE)

    text = re.sub(r"\b(hi|hello)\s*,?\s+(hi|hello)\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,+", ", ", text)
    return re.sub(r"\s+", " ", text).strip(" ,")


def _ensure_minimum_plan_questions(questions: list, job_role: str, min_count: int = 12) -> list:
    """Prevent short LLM responses from ending the interview after only 1-2 questions."""
    fallback = [
        ("technical", "core skills", f"What are the most important skills you use for a {job_role} role, and how have you applied them?"),
        ("technical", "problem solving", f"Walk me through a challenging {job_role} problem you solved recently."),
        ("technical", "debugging", "How do you approach debugging when something works locally but fails in production?"),
        ("technical", "architecture", f"How would you design a reliable solution for a common {job_role} workflow?"),
        ("technical", "tools", f"Which tools or frameworks are strongest for your {job_role} work, and why?"),
        ("situational", "prioritization", "If you had two urgent tasks and limited time, how would you decide what to do first?"),
        ("situational", "ambiguity", "Tell me how you handle unclear requirements from a manager or client."),
        ("hr", "teamwork", "Tell me about a time you worked with a difficult teammate and how you handled it."),
        ("hr", "ownership", "Give me an example of a time you took ownership without being asked."),
        ("cultural", "motivation", f"Why are you interested in this {job_role} path right now?"),
        ("hr", "availability", "What is your current availability or notice period?"),
        ("closing", "candidate questions", "Before we wrap up, do you have any questions for me about the role or team?"),
    ]

    existing_text = {str(q.get("text", "")).strip().lower() for q in questions}
    for q_type, category, text in fallback:
        if len(questions) >= min_count:
            break
        if text.lower() in existing_text:
            continue
        questions.append({
            "id": len(questions) + 1,
            "text": text,
            "type": q_type,
            "category": category,
            "follow_up_hint": "Ask for a concrete example, tradeoff, or measurable outcome if the answer is vague.",
        })
        existing_text.add(text.lower())

    for idx, question in enumerate(questions):
        question["id"] = idx + 1
    return questions


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
    result["overall_score"] = _clamp_score(round(weighted))

    overall = result["overall_score"]
    result["grade"] = (
        "A+" if overall >= 95 else
        "A" if overall >= 90 else
        "B+" if overall >= 80 else
        "B" if overall >= 70 else
        "C" if overall >= 60 else
        "D"
    )
    result["verdict"] = (
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
