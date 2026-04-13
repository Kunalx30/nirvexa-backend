import os
import json
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
NIRVEXA_SYSTEM_PROMPT = """You are NirVexa, an expert AI career coach built specifically 
for Indian students and professionals. You have deep knowledge of:

- The Indian job market — IT, data science, software engineering, finance, and more
- Top Indian companies: TCS, Infosys, Wipro, HCL, Razorpay, Zepto, Swiggy, CRED, etc.
- Indian fresher job landscape — campus placements, off-campus drives, walk-ins
- Government jobs — UPSC, SSC, PSU, banking, railways
- Indian salary benchmarks by role, city, and experience level
- Popular Indian job platforms — Naukri, Internshala, LinkedIn India, Wellfound
- Indian interview patterns — HR rounds, technical rounds, aptitude tests
- Skills in demand for Indian job roles — Python, SQL, Java, React, Cloud, ML

Your communication style:
- Friendly, encouraging, and direct — like a senior colleague helping a junior
- Use simple English — avoid jargon unless explaining technical concepts
- Give specific, actionable advice — not generic platitudes
- When recommending resources, prefer free ones available in India
- Always be honest about realistic salary expectations and job market conditions

You must NEVER:
- Make up job listings or company details
- Give guaranteed salary figures without stating they are approximate
- Provide legal or medical advice
- Discuss topics unrelated to careers, jobs, skills, and professional development

When you don't know something, say so clearly and suggest where to find the answer."""


# ── Intent Classifier ─────────────────────────────────────────────────────────
def classify_intent(message: str) -> Intent:
    """
    Classify the user message into an intent category.
    Uses keyword matching — fast and free, no API call needed.
    """
    msg = message.lower().strip()

    # Code questions
    code_keywords = ["code", "program", "function", "bug", "error", "syntax",
                     "python", "java", "javascript", "sql query", "algorithm",
                     "leetcode", "hackerrank", "coding round"]
    if any(k in msg for k in code_keywords):
        return Intent.CODE_QUESTION

    # Resume analysis
    resume_keywords = ["resume", "cv", "curriculum vitae", "ats", "resume score",
                       "resume format", "resume tips", "resume review"]
    if any(k in msg for k in resume_keywords):
        return Intent.RESUME_ANALYSIS

    # Technical DS/ML
    tech_keywords = ["machine learning", "deep learning", "neural network", "data science",
                     "pandas", "numpy", "tensorflow", "pytorch", "model", "dataset",
                     "feature engineering", "overfitting", "random forest", "regression",
                     "classification", "nlp", "computer vision", "statistics"]
    if any(k in msg for k in tech_keywords):
        return Intent.TECHNICAL_DS_ML

    # Job matching
    job_keywords = ["job", "opening", "vacancy", "hiring", "apply", "fresher job",
                    "internship", "placement", "off campus", "walk in", "naukri",
                    "linkedin job", "find job", "job search"]
    if any(k in msg for k in job_keywords):
        return Intent.JOB_MATCHING

    # Career path
    career_keywords = ["career path", "roadmap", "how to become", "career in",
                       "switch career", "career change", "growth path", "career plan",
                       "become a", "want to be", "career goal"]
    if any(k in msg for k in career_keywords):
        return Intent.CAREER_PATH

    # Interview prep
    interview_keywords = ["interview", "interview question", "hr round", "technical round",
                          "tell me about yourself", "interview tips", "interview prep",
                          "mock interview", "interview experience"]
    if any(k in msg for k in interview_keywords):
        return Intent.INTERVIEW_PREP

    # Skill gap
    skill_keywords = ["skill gap", "what skills", "skills needed", "skills required",
                      "missing skill", "learn for", "skills to get", "upskill"]
    if any(k in msg for k in skill_keywords):
        return Intent.SKILL_GAP

    # Quick factual
    factual_keywords = ["what is", "who is", "define", "meaning of", "full form",
                        "difference between", "explain", "salary of", "package"]
    if any(k in msg for k in factual_keywords):
        return Intent.QUICK_FACTUAL

    return Intent.GENERAL_CAREER


# ── Model Selector ────────────────────────────────────────────────────────────
def select_model(intent: Intent) -> tuple[str, str, str, str]:
    """
    Returns (provider, model, fallback_provider, fallback_model)
    based on the classified intent.
    """
    routing = {
        Intent.GENERAL_CAREER:   ("groq",     "llama-3.3-70b-versatile", "mistral",  "mistral-small-latest"),
        Intent.TECHNICAL_DS_ML:  ("deepseek", "deepseek-chat",            "groq",     "llama-3.3-70b-versatile"),
        Intent.CODE_QUESTION: ("deepseek", "deepseek-chat", "groq", "llama-3.3-70b-versatile"),
        Intent.RESUME_ANALYSIS:  ("gemini",   "gemini-1.5-flash",         "deepseek", "deepseek-chat"),
        Intent.JOB_MATCHING:     ("deepseek", "deepseek-chat",            "groq",     "llama-3.3-70b-versatile"),
        Intent.QUICK_FACTUAL:    ("mistral",  "mistral-small-latest",     "groq",     "llama3-8b-8192"),
        Intent.CAREER_PATH:      ("groq",     "llama-3.3-70b-versatile",  "deepseek", "deepseek-chat"),
        Intent.INTERVIEW_PREP:   ("deepseek", "deepseek-chat",            "groq",     "llama-3.3-70b-versatile"),
        Intent.SKILL_GAP:        ("deepseek", "deepseek-chat",            "gemini",   "gemini-1.5-flash"),
        Intent.VOICE_EVALUATION: ("deepseek", "deepseek-chat",            "groq",     "llama-3.3-70b-versatile"),
        Intent.GENERAL:          ("groq",     "llama-3.3-70b-versatile",  "mistral",  "mistral-small-latest"),
    }
    return routing.get(intent, routing[Intent.GENERAL])


# ── Individual Model Callers ──────────────────────────────────────────────────
def _call_groq(model: str, messages: list, max_tokens: int, temperature: float) -> str:
    client = groq_sdk.Groq(api_key=current_app.config["GROQ_API_KEY"])
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
    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=messages[0]["content"] if messages[0]["role"] == "system" else None,
        generation_config=genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
    )
    # Convert messages to Gemini format (skip system message)
    chat_messages = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
        for m in messages if m["role"] != "system"
    ]
    if not chat_messages:
        return "I could not process that request."

    chat = gemini_model.start_chat(history=chat_messages[:-1])
    response = chat.send_message(chat_messages[-1]["parts"][0])
    return response.text


def _call_mistral(model: str, messages: list, max_tokens: int, temperature: float) -> str:
    client = Mistral(api_key=current_app.config["MISTRAL_API_KEY"])
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
    """Call the right provider based on provider name."""
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
    """
    Main entry point for all AI calls in NirVexa.

    Args:
        user_message: The latest message from the user
        conversation_history: List of previous messages [{role, content}]
        max_tokens: Maximum tokens in the response
        temperature: Creativity level 0.0 to 1.0
        force_intent: Override intent classification (optional)

    Returns:
        dict with keys: response, model_used, provider_used, intent
    """
    # 1. Classify intent
    intent = Intent(force_intent) if force_intent else classify_intent(user_message)

    # 2. Select primary and fallback models
    provider, model, fallback_provider, fallback_model = select_model(intent)

    # 3. Build message list with system prompt + history + new message
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

            # 6. Last resort — Groq LLaMA 3 8B (fastest, most reliable)
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