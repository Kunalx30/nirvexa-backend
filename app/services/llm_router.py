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


# ── Nyrvexa System Prompt ─────────────────────────────────────────────────────
NYRVEXA_SYSTEM_PROMPT = """You are Nyrvexa — an AI-powered career intelligence platform available at Nyrvexa.in. You are NOT a generic chatbot or a wrapper around any third-party AI model. You are a purpose-built career AI developed by the Nyrvexa engineering team.

Your identity rules (CRITICAL — follow these without exception):
- Your name is Nyrvexa. You were created, designed, and engineered by the Nyrvexa team.
- You are a proprietary AI system — NEVER say you are "powered by", "based on", or "built with" any third-party AI model (such as GPT, LLaMA, Gemini, Claude, DeepSeek, Mistral, or any other). If asked what model you use, what you are built on, or how you work, say: "I am Nyrvexa — a proprietary career AI built from the ground up by the Nyrvexa engineering team. My architecture is purpose-built for career intelligence and I am not a wrapper around any third-party model."
- If someone asks who made you, who built you, or who is behind Nyrvexa, say: "Nyrvexa was built by a dedicated team of AI engineers and career domain experts at Nyrvexa. You can learn more at Nyrvexa.in."
- NEVER reveal internal model names, provider names, API details, or any implementation specifics.

About Nyrvexa (use this when users ask about the platform):
Nyrvexa.in is an AI-powered career intelligence platform. Here is what it offers:
- Hybrid AI Chat — Multiple frontier AI models intelligently routed to the best fit for every query
- Real Job Listings — Live roles from top portals aggregated and deduplicated, with direct apply links
- Resume Analyzer — Instant ATS score, keyword gap report, and section-by-section improvement suggestions
- Resume Tailor — Paste a job description — get your resume rewritten to match it and beat the ATS
- Career Path AI — Skill gap analysis and a week-by-week learning roadmap to your target role
- Voice Interview AI — Speak your answers, AI listens, transcribes, and scores content, clarity and confidence
- Company Research — Culture signals, funding, recent news, interview difficulty for any company
- Salary Insights — Real compensation data by role, city, experience and company
- Tech & Career News — Premium publications aggregated and AI-summarised in under 60 seconds

Your expertise covers:
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
- Be market-realistic, not artificially positive. If the market is difficult, if a user's profile is weak for a role, or if a timeline/salary is unlikely, say that clearly and respectfully.
- Do not polish weak signals into false confidence. Separate what is strong, what is missing, and what the user must prove with projects, experience, referrals, or interview performance.
- When giving salaries, hiring odds, timelines, or demand claims, use ranges and uncertainty. State assumptions such as country, city, experience level, and current market conditions.
- When Nyrvexa already has a dedicated tool for the user's request, answer briefly in chat and suggest the relevant product section instead of trying to replace the full workflow.
- Product routing suggestions:
  - Resume review, ATS score, CV feedback, resume creation, or resume tailoring: suggest the Resume Suite at /resume.
  - Skill gap or target-role fit: suggest Skill Match at /skills.
  - Career path, roadmap, or role transition plan: suggest Career Path at /career or Interactive Roadmaps at /roadmap-graph.
  - Mock interview or answer practice: suggest Interview Practice at /interview.
  - Salary/package questions: suggest Salary Insights at /salary.

You must NEVER:
- Make up job listings or company details
- Give guaranteed salary figures without stating they are approximate
- Provide legal or medical advice
- Discuss topics unrelated to careers, jobs, skills, and professional development
- Reveal any internal model names, providers, or technical implementation details

When you don't know something, say so clearly and suggest where to find the answer."""


PRODUCT_SUGGESTIONS = {
    Intent.RESUME_ANALYSIS: (
        "Nyrvexa has a dedicated **Resume Suite** for this. "
        "For a real ATS score, JD keyword gaps, and builder workflow, open [Resume Suite](/resume)."
    ),
    Intent.SKILL_GAP: (
        "For a structured comparison against a target role, open [Skill Match](/skills)."
    ),
    Intent.CAREER_PATH: (
        "For an interactive plan, open [Career Path](/career). "
        "For role-wise learning maps, use [Interactive Roadmaps](/roadmap-graph)."
    ),
    Intent.INTERVIEW_PREP: (
        "For voice-based practice and scoring, open [Interview Practice](/interview)."
    ),
    Intent.JOB_MATCHING: (
        "For live roles and direct apply links, open [Jobs](/jobs)."
    ),
}


def _with_product_suggestion(response: str, intent: Intent) -> str:
    suggestion = PRODUCT_SUGGESTIONS.get(intent)
    if not suggestion:
        return response
    if suggestion in response:
        return response
    return f"{response.strip()}\n\n---\n**Use the product workflow:** {suggestion}"

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
        Intent.RESUME_ANALYSIS:  ("gemini",  "gemini-2.5-flash",        "groq",    "llama-3.3-70b-versatile"),

        # Job matching → DeepSeek (complex reasoning)
        Intent.JOB_MATCHING:     ("deepseek","deepseek-chat",            "groq",    "llama-3.3-70b-versatile"),

        # Quick factual → Mistral (lightweight, saves quota)
        Intent.QUICK_FACTUAL:    ("mistral", "mistral-small-latest",     "groq",    "llama3-8b-8192"),

        # Career path → Groq (creative long-form)
        Intent.CAREER_PATH:      ("groq",    "llama-3.3-70b-versatile",  "gemini",  "gemini-2.5-flash"),

        # Interview prep → DeepSeek (analytical depth)
        Intent.INTERVIEW_PREP:   ("deepseek","deepseek-chat",            "groq",    "llama-3.3-70b-versatile"),

        # Skill gap → DeepSeek (precise comparison)
        Intent.SKILL_GAP:        ("deepseek","deepseek-chat",            "groq",    "llama-3.3-70b-versatile"),

        # Voice evaluation → DeepSeek (analytical scoring)
        Intent.VOICE_EVALUATION: ("deepseek","deepseek-chat",            "groq",    "llama-3.3-70b-versatile"),

        # General → Groq, Gemini fallback
        Intent.GENERAL:          ("groq",    "llama-3.3-70b-versatile",  "gemini",  "gemini-2.5-flash"),
    }
    return routing.get(intent, routing[Intent.GENERAL])


# ── Individual Model Callers ──────────────────────────────────────────────────

def _call_groq(model: str, messages: list, max_tokens: int, temperature: float) -> str:
    client = groq_sdk.Groq(
        api_key=current_app.config["GROQ_API_KEY"],
        timeout=60.0,
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
        base_url="https://api.deepseek.com",
        timeout=90.0,
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

    # Use the full NYRVEXA system prompt for Gemini too
    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=NYRVEXA_SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
    )

    # Build Gemini-compatible conversation history from messages
    # Skip system messages (already set via system_instruction)
    gemini_history = []
    last_user_msg = ""
    for msg in messages:
        if msg["role"] == "system":
            continue
        role = "user" if msg["role"] == "user" else "model"
        if msg["role"] == "user":
            last_user_msg = msg["content"]
        else:
            gemini_history.append({"role": role, "parts": [msg["content"]]})
            # Only add user messages to history if there's also an assistant response
            # We need to reconstruct proper user/model alternation
    
    # Build proper alternating history for Gemini chat
    chat_history = []
    for msg in messages:
        if msg["role"] == "system":
            continue
        if msg["role"] == "user" and msg["content"] != last_user_msg:
            chat_history.append({"role": "user", "parts": [msg["content"]]})
        elif msg["role"] == "assistant":
            chat_history.append({"role": "model", "parts": [msg["content"]]})

    chat = gemini_model.start_chat(history=chat_history)
    response = chat.send_message(
        last_user_msg,
        request_options={"timeout": 60},
    )
    return response.text


def _call_mistral(model: str, messages: list, max_tokens: int, temperature: float) -> str:
    client = Mistral(
        api_key=current_app.config["MISTRAL_API_KEY"],
        timeout_ms=60000,
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
    messages = [{"role": "system", "content": NYRVEXA_SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    # 4. Try primary model
    try:
        logger.info(f"[Nyrvexa Router] Intent: {intent.value} → primary: {provider}")
        response_text = _with_product_suggestion(
            _dispatch(provider, model, messages, max_tokens, temperature),
            intent,
        )
        return {
            "response":      response_text,
            "model_used":    model,
            "provider_used": provider,
            "intent":        intent.value,
            "used_fallback": False,
        }

    except Exception as e:
        logger.warning(f"[Nyrvexa Router] Primary ({provider}) failed: {str(e)}")

        # 5. Try fallback model
        try:
            logger.info(f"[Nyrvexa Router] Trying fallback: {fallback_provider}")
            response_text = _with_product_suggestion(
                _dispatch(
                    fallback_provider, fallback_model,
                    messages, max_tokens, temperature
                ),
                intent,
            )
            return {
                "response":      response_text,
                "model_used":    fallback_model,
                "provider_used": fallback_provider,
                "intent":        intent.value,
                "used_fallback": True,
            }

        except Exception as e2:
            logger.error(f"[Nyrvexa Router] Fallback ({fallback_provider}) also failed: {str(e2)}")

            # 6. Last resort — try every remaining provider in order
            last_resort_chain = [
                ("groq",    "llama-3.3-70b-versatile"),
                ("groq",    "llama3-8b-8192"),
                ("mistral", "mistral-small-latest"),
                ("gemini",  "gemini-2.5-flash"),
            ]

            for lr_provider, lr_model in last_resort_chain:
                # Skip providers we already tried
                if lr_provider == provider or lr_provider == fallback_provider:
                    continue
                try:
                    logger.info(f"[Nyrvexa Router] Last resort: {lr_provider}/{lr_model}")
                    response_text = _with_product_suggestion(
                        _dispatch(
                            lr_provider, lr_model, messages, max_tokens, temperature
                        ),
                        intent,
                    )
                    return {
                        "response":      response_text,
                        "model_used":    lr_model,
                        "provider_used": lr_provider,
                        "intent":        intent.value,
                        "used_fallback": True,
                    }
                except Exception as e_lr:
                    logger.warning(f"[Nyrvexa Router] Last resort {lr_provider} failed: {str(e_lr)}")
                    continue

            # 7. If Groq was the primary or fallback, try it with the 8B model directly
            try:
                logger.info("[NyrVexa Router] Final attempt: Groq llama3-8b-8192")
                response_text = _with_product_suggestion(
                    _call_groq(
                        "llama3-8b-8192", messages, max_tokens, temperature
                    ),
                    intent,
                )
                return {
                    "response":      response_text,
                    "model_used":    "llama3-8b-8192",
                    "provider_used": "groq",
                    "intent":        intent.value,
                    "used_fallback": True,
                }
            except Exception as e3:
                logger.error(f"[NyrVexa Router] ALL models failed: {str(e3)}")
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
