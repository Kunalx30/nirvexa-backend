"""
app/services/interview_service.py
NirVexa — Phase 6.6: Text Interview Prep Service
"""

import logging
import json
import openai
from flask import current_app

logger = logging.getLogger(__name__)


def generate_questions(role: str, prep_type: str) -> dict:
    """
    Generate 10 interview questions for the given role and type using DeepSeek V3.
    prep_type: 'hr' or 'technical'
    """

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
        client = openai.OpenAI(
            api_key=current_app.config["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
            timeout=60.0
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are an interview coach. Always respond with valid JSON only. No markdown, no backticks."
                },
                {
                    "role": "user",
                    "content": prompt
                }
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

        result = json.loads(raw)
        return {"success": True, "questions": result}

    except json.JSONDecodeError as e:
        logger.error("[InterviewService] JSON parse failed in generate_questions: %s", e)
        return {"success": False, "error": "AI returned malformed response. Please try again."}

    except Exception as e:
        logger.error("[InterviewService] generate_questions failed: %s", e)
        return {"success": False, "error": "Failed to generate questions. Please try again."}


def evaluate_answer(question: str, answer: str, role: str) -> dict:
    """
    Evaluate a written interview answer using DeepSeek V3.
    Returns score, feedback, suggested answer, and keywords missed.
    """

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
- grade must follow standard grading: A+ (95+), A (90+), B+ (80+), B (70+), C (60+), D (below 60)
- feedback must be specific to THIS answer, not generic
- suggested_answer must be role-specific and concise
- verdict: Strong if score >= 80, Acceptable if 60-79, Needs Work if below 60
- keywords_missed should be important terms/concepts the candidate should have mentioned"""

    try:
        client = openai.OpenAI(
            api_key=current_app.config["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
            timeout=60.0
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are an interview evaluator. Always respond with valid JSON only. No markdown, no backticks."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1024,
            temperature=0.3,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)
        return {"success": True, "data": result}

    except json.JSONDecodeError as e:
        logger.error("[InterviewService] JSON parse failed in evaluate_answer: %s", e)
        return {"success": False, "error": "AI returned malformed response. Please try again."}

    except Exception as e:
        logger.error("[InterviewService] evaluate_answer failed: %s", e)
        return {"success": False, "error": "Failed to evaluate answer. Please try again."}