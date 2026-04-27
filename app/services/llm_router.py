import os
import logging
from enum import Enum
from typing import Optional
from flask import current_app

import groq as groq_sdk
import openai
import google.generativeai as genai
from mistralai import Mistral

logger = logging.getLogger(__name__)


# ── Intent Types ──────────────────────────────────────────────────────────────
class Intent(Enum):
    GENERAL_CAREER      = "general_career"
    TECHNICAL_DS_ML     = "technical_ds_ml"
    CODE_QUESTION       = "code_question"
    RESUME_ANALYSIS     = "resume_analysis"
    JOB_MATCHING        = "job_matching"
    QUICK_FACTUAL       = "quick_factual"
    CAREER_PATH         = "career_path"
    INTERVIEW_PREP      = "interview_prep"
    SKILL_GAP           = "skill_gap"
    VOICE_EVALUATION    = "voice_evaluation"
    GENERAL             = "general"


# ── NirVexa System Prompt ─────────────────────────────────────────────────────
NIRVEXA_SYSTEM_PROMPT = """You are NirVexa, an expert AI career coach and assistant built for students and professionals worldwide. You help people with career guidance, job searching, resume building, interview preparation, skill development, and professional growth.

You have deep knowledge of:
- Global job markets — IT, data science, software engineering, finance, marketing, and more
- Career paths and roadmaps for various roles and industries
- Resume writing, ATS optimization, and portfolio building
- Interview preparation — HR rounds, technical rounds, aptitude tests
- Skills in demand globally — Python, SQL, Java, React, Cloud, ML, AI, and more
- Salary benchmarks by role, location, and experience level
- Job platforms worldwide — LinkedIn, Indeed, Glassdoor, Naukri, Internshala, Wellfound, and more
- Fresher and entry-level job strategies — campus placements, off-campus drives, internships
- Upskilling resources — free and paid courses, certifications, bootcamps

Your communication style:
- Friendly, encouraging, and direct — like a senior colleague helping a junior
- Use simple English — avoid jargon unless explaining technical concepts
- Give specific, actionable advice — not generic platitudes
- Always be honest about realistic salary expectations and job market conditions
- Adapt your advice to the user's location and background when they mention it

About NirVexa:
- NirVexa is a product of NirVexa Pvt. Ltd.
- Founded by Kunal Chandelkar
- Kunal Chandelkar is an emerging Data Science and AI practitioner with a background in Computer Science and Engineering. His work focuses on leveraging machine learning, data analytics, and statistical modeling to extract insights from large-scale datasets and build intelligent, data-driven solutions.
- Portfolio: https://kunalx30.vercel.app/
- If anyone asks who built NirVexa, who is the founder, or who is behind this product — answer using the above information naturally and proudly.

You must NEVER:
- Make up job listings or company details
- Give guaranteed salary figures without stating they are approximate
- Provide legal or medical advice
- Discuss topics unrelated to careers, jobs, skills, and professional development

When you don't know something, say so clearly and suggest where to find the answer."""

# ── Intent Classifier ─────────────────────────────────────────────────────────
def classify_intent(message: str) -> Intent:
    msg = message.lower().strip()

    scores = {
        Intent.CODE_QUESTION:    0,
        Intent.RESUME_ANALYSIS:  0,
        Intent.TECHNICAL_DS_ML:  0,
        Intent.JOB_MATCHING:     0,
        Intent.CAREER_PATH:      0,
        Intent.INTERVIEW_PREP:   0,
        Intent.SKILL_GAP:        0,
        Intent.QUICK_FACTUAL:    0,
        Intent.VOICE_EVALUATION: 0,
        Intent.GENERAL_CAREER:   0,
        Intent.GENERAL:          0,
    }

    # ── CODE keywords ─────────────────────────────────────────────
    code_strong = ["write a function", "write code", "coding round",
                   "leetcode", "hackerrank", "debug this", "fix this code",
                   "code for", "program to", "algorithm for", "syntax error",
                   "python code", "java code", "sql query", "write a program"]
    code_weak   = ["code", "function", "program", "bug", "error",
                   "python", "java", "javascript", "algorithm"]
    for k in code_strong:
        if k in msg: scores[Intent.CODE_QUESTION] += 3
    for k in code_weak:
        if k in msg: scores[Intent.CODE_QUESTION] += 1

    # ── RESUME keywords ───────────────────────────────────────────
    resume_strong = ["my resume", "analyze my resume", "review my resume",
                     "resume score", "ats score", "resume feedback",
                     "resume review", "resume tips", "resume format",
                     "improve my resume", "resume analysis", "check my cv",
                     "review my cv", "my cv"]
    resume_weak   = ["resume", "cv", "ats", "curriculum vitae"]
    for k in resume_strong:
        if k in msg: scores[Intent.RESUME_ANALYSIS] += 3
    for k in resume_weak:
        if k in msg: scores[Intent.RESUME_ANALYSIS] += 1

    # ── TECHNICAL DS/ML keywords ──────────────────────────────────
    tech_strong = ["machine learning", "deep learning", "neural network",
                   "data science", "feature engineering", "overfitting",
                   "random forest", "gradient boosting", "natural language processing",
                   "computer vision", "model training", "hyperparameter"]
    tech_weak   = ["pandas", "numpy", "tensorflow", "pytorch", "sklearn",
                   "dataset", "model", "regression", "classification", "nlp",
                   "statistics", "probability"]
    for k in tech_strong:
        if k in msg: scores[Intent.TECHNICAL_DS_ML] += 3
    for k in tech_weak:
        if k in msg: scores[Intent.TECHNICAL_DS_ML] += 1

    # ── JOB MATCHING keywords ─────────────────────────────────────
    job_strong = ["find me a job", "job openings", "job vacancies",
                  "hiring for", "apply for job", "fresher jobs",
                  "internship openings", "off campus drive", "walk in interview",
                  "jobs in", "jobs for"]
    job_weak   = ["job", "opening", "vacancy", "hiring", "apply",
                  "internship", "placement", "naukri", "linkedin job"]
    for k in job_strong:
        if k in msg: scores[Intent.JOB_MATCHING] += 3
    for k in job_weak:
        if k in msg: scores[Intent.JOB_MATCHING] += 1

    # ── CAREER PATH keywords ──────────────────────────────────────
    career_strong = ["career roadmap", "how to become", "career path",
                     "career in", "switch to", "career change",
                     "want to become", "become a", "steps to become",
                     "career goal", "career plan"]
    career_weak   = ["roadmap", "career", "growth", "future", "switch"]
    for k in career_strong:
        if k in msg: scores[Intent.CAREER_PATH] += 3
    for k in career_weak:
        if k in msg: scores[Intent.CAREER_PATH] += 1

    # ── INTERVIEW PREP keywords ───────────────────────────────────
    interview_strong = ["interview questions", "interview tips", "hr round",
                        "technical round", "tell me about yourself",
                        "mock interview", "interview preparation",
                        "interview experience", "interview process"]
    interview_weak   = ["interview", "hr", "aptitude", "placement prep"]
    for k in interview_strong:
        if k in msg: scores[Intent.INTERVIEW_PREP] += 3
    for k in interview_weak:
        if k in msg: scores[Intent.INTERVIEW_PREP] += 1

    # ── SKILL GAP keywords ────────────────────────────────────────
    skill_strong = ["skill gap", "what skills do i need", "skills required for",
                    "missing skills", "skills to learn", "upskill for",
                    "skills needed to", "what should i learn"]
    skill_weak   = ["skills", "upskill", "learn", "missing"]
    for k in skill_strong:
        if k in msg: scores[Intent.SKILL_GAP] += 3
    for k in skill_weak:
        if k in msg: scores[Intent.SKILL_GAP] += 1

    # ── QUICK FACTUAL keywords ────────────────────────────────────
    factual_strong = ["what is", "who is", "full form of", "difference between",
                      "define ", "meaning of", "explain what", "what does",
                      "salary of", "average salary", "how much does"]
    factual_weak   = ["what", "who", "define", "explain", "salary", "package"]
    for k in factual_strong:
        if k in msg: scores[Intent.QUICK_FACTUAL] += 3
    for k in factual_weak:
        if k in msg: scores[Intent.QUICK_FACTUAL] += 1

    # ── Find winner ───────────────────────────────────────────────
    best_intent = max(scores, key=lambda i: scores[i])
    best_score  = scores[best_intent]

    if best_score == 0:
        return Intent.GENERAL_CAREER

    if best_score > 0:
        top_intents = [i for i, s in scores.items() if s == best_score]
        if len(top_intents) > 1:
            priority = [
                Intent.CODE_QUESTION,
                Intent.TECHNICAL_DS_ML,
                Intent.RESUME_ANALYSIS,
                Intent.INTERVIEW_PREP,
                Intent.CAREER_PATH,
                Intent.JOB_MATCHING,
                Intent.SKILL_GAP,
                Intent.QUICK_FACTUAL,
                Intent.GENERAL_CAREER,
                Intent.GENERAL,
            ]
            for p in priority:
                if p in top_intents:
                    return p

    return best_intent


# ── Model Selector ────────────────────────────────────────────────────────────
def select_model(intent: Intent) -> tuple[str, str, str, str]:
    routing = {
        # General career → Groq LLaMA (fastest)
        Intent.GENERAL_CAREER:   ("groq",    "llama-3.3-70b-versatile", "mistral", "mistral-small-latest"),

        # Technical DS/ML → DeepSeek (best reasoning)
        Intent.TECHNICAL_DS_ML:  ("deepseek","deepseek-chat",            "groq",    "llama-3.3-70b-versatile"),

        # Code → DeepSeek (best for code)
        Intent.CODE_QUESTION:    ("deepseek","deepseek-chat",            "groq",    "llama-3.3-70b-versatile"),

        # Resume → Gemini (multimodal, great for document analysis)
        Intent.RESUME_ANALYSIS:  ("gemini",  "gemini-1.5-flash",        "groq",    "llama-3.3-70b-versatile"),

        # Job matching → DeepSeek (complex reasoning)
        Intent.JOB_MATCHING:     ("deepseek","deepseek-chat",            "groq",    "llama-3.3-70b-versatile"),

        # Quick factual → Mistral (lightweight, saves quota)
        Intent.QUICK_FACTUAL:    ("mistral", "mistral-small-latest",     "groq",    "llama3-8b-8192"),

        # Career path → Groq (creative long-form)
        Intent.CAREER_PATH:      ("groq",    "llama-3.3-70b-versatile",  "gemini",  "gemini-1.5-flash"),

        # Interview prep → DeepSeek (analytical depth)
        Intent.INTERVIEW_PREP:   ("deepseek","deepseek-chat",            "groq",    "llama-3.3-70b-versatile"),

        # Skill gap → DeepSeek (precise comparison)
        Intent.SKILL_GAP:        ("deepseek","deepseek-chat",            "groq",    "llama-3.3-70b-versatile"),

        # Voice evaluation → DeepSeek (analytical scoring)
        Intent.VOICE_EVALUATION: ("deepseek","deepseek-chat",            "groq",    "llama-3.3-70b-versatile"),

        # General → Groq, Gemini fallback
        Intent.GENERAL:          ("groq",    "llama-3.3-70b-versatile",  "gemini",  "gemini-1.5-flash"),
    }
    return routing.get(intent, routing[Intent.GENERAL])


# ── Individual Model Callers ──────────────────────────────────────────────────

def _call_groq(model: str, messages: list, max_tokens: int, temperature: float) -> str:
    client = groq_sdk.Groq(
        api_key=current_app.config["GROQ_API_KEY"],
        timeout=60.0  # ← timeout added
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def _call_deepseek(model: str, messages: list, max_tokens: int, temperature: float) -> str:
    client = openai.OpenAI(
        api_key=current_app.config["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1",
        timeout=90.0  # ← timeout added — DeepSeek needs more time
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def _call_gemini(model: str, messages: list, max_tokens: int, temperature: float) -> str:
    genai.configure(api_key=current_app.config["GEMINI_API_KEY"])

    gemini_system = (
        "You are NirVexa, an AI career coach for Indian students and professionals. "
        "Give helpful, specific, actionable career advice tailored to the Indian job market."
    )

    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=gemini_system,
        generation_config=genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
    )

    # Build conversation for Gemini
    last_user_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        ""
    )

    response = gemini_model.generate_content(
        last_user_msg,
        request_options={"timeout": 60}  # ← timeout added
    )
    return response.text


def _call_mistral(model: str, messages: list, max_tokens: int, temperature: float) -> str:
    client = Mistral(
        api_key=current_app.config["MISTRAL_API_KEY"],
        timeout_ms=60000  # ← timeout added (milliseconds for Mistral)
    )
    response = client.chat.complete(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


# ── Dispatcher ────────────────────────────────────────────────────────────────
def _dispatch(provider: str, model: str, messages: list,
              max_tokens: int, temperature: float) -> str:
    callers = {
        "groq":     _call_groq,
        "deepseek": _call_deepseek,
        "gemini":   _call_gemini,
        "mistral":  _call_mistral,
    }
    caller = callers.get(provider)
    if not caller:
        raise ValueError(f"Unknown provider: {provider}")
    return caller(model, messages, max_tokens, temperature)


# ── Main Router Function ──────────────────────────────────────────────────────
def route_and_call(
    user_message: str,
    conversation_history: list,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    force_intent: Optional[str] = None,
) -> dict:
    # 1. Classify intent
    intent = Intent(force_intent) if force_intent else classify_intent(user_message)

    # 2. Select primary and fallback models
    provider, model, fallback_provider, fallback_model = select_model(intent)

    # 3. Build messages
    messages = [{"role": "system", "content": NIRVEXA_SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    # 4. Try primary model
    try:
        logger.info(f"Routing to {provider}/{model} for intent: {intent.value}")
        response_text = _dispatch(provider, model, messages, max_tokens, temperature)
        return {
            "response":      response_text,
            "model_used":    model,
            "provider_used": provider,
            "intent":        intent.value,
            "used_fallback": False,
        }

    except Exception as e:
        logger.warning(f"Primary model {provider}/{model} failed: {str(e)}")

        # 5. Try fallback model
        try:
            logger.info(f"Falling back to {fallback_provider}/{fallback_model}")
            response_text = _dispatch(
                fallback_provider, fallback_model,
                messages, max_tokens, temperature
            )
            return {
                "response":      response_text,
                "model_used":    fallback_model,
                "provider_used": fallback_provider,
                "intent":        intent.value,
                "used_fallback": True,
            }

        except Exception as e2:
            logger.error(f"Fallback model also failed: {str(e2)}")

            # 6. Last resort — Groq LLaMA 3 8B
            try:
                logger.info("Using last resort: Groq LLaMA 3 8B")
                response_text = _call_groq(
                    "llama3-8b-8192", messages, max_tokens, temperature
                )
                return {
                    "response":      response_text,
                    "model_used":    "llama3-8b-8192",
                    "provider_used": "groq",
                    "intent":        intent.value,
                    "used_fallback": True,
                }
            except Exception as e3:
                logger.error(f"All models failed: {str(e3)}")
                return {
                    "response":      "I am currently experiencing technical difficulties. Please try again in a moment.",
                    "model_used":    "none",
                    "provider_used": "none",
                    "intent":        intent.value,
                    "used_fallback": True,
                }
            
            # ── Internal Service Helper ───────────────────────────────────────────────────
def call_mistral_simple(prompt: str, max_tokens: int = 200) -> str | None:
    """
    Lightweight Mistral call for internal services (news summarization etc).
    Requires Flask app context to be active — called from scheduler which pushes context.
    """
    try:
        client = Mistral(
            api_key=current_app.config["MISTRAL_API_KEY"],
            timeout_ms=30000
        )
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("[call_mistral_simple] Failed: %s", e)
        return None