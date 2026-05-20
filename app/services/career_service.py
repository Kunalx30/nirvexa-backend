import logging
import json
from flask import current_app
import groq as groq_sdk

logger = logging.getLogger(__name__)


def generate_career_path(
    current_role: str,
    target_role: str,
    current_skills: list,
    experience_years: int
) -> dict:
    """
    Calls Groq LLaMA 3.3 70B to generate a detailed career roadmap.
    Returns structured dict with steps, skill gap, and estimated time.
    """

    skills_str = ", ".join(current_skills) if current_skills else "None mentioned"

    prompt = f"""You are an expert career advisor for the Indian job market.

A professional wants to transition their career:
- Current Role: {current_role}
- Target Role: {target_role}
- Current Skills: {skills_str}
- Years of Experience: {experience_years}

Generate a detailed, realistic career roadmap tailored to the Indian job market.

Respond ONLY with a valid JSON object. No preamble, no markdown, no explanation outside the JSON.

The JSON must follow this exact structure:
{{
  "steps": [
    {{
      "step_number": 1,
      "title": "short title of this step",
      "skill_to_learn": "specific skill or topic",
      "resources": ["resource 1", "resource 2", "resource 3"],
      "estimated_weeks": 4,
      "why_important": "one sentence explanation"
    }}
  ],
  "skill_gap": {{
    "missing": ["skill1", "skill2"],
    "already_have": ["skill3", "skill4"]
  }},
  "estimated_total_weeks": 24,
  "difficulty": "Beginner | Intermediate | Advanced",
  "summary": "2-3 sentence overall summary of the transition"
}}

Rules:
- Generate 5 to 8 steps
- For resources, do NOT hallucinate broken direct links. You MUST mix and match EXACT, WORKING search URLs from these reliable, multi-domain platforms:
  * YouTube Video: "YouTube: https://www.youtube.com/results?search_query=learn+[skill_name]"
  * Coursera Courses: "Coursera: https://www.coursera.org/search?query=[skill_name]"
  * Udemy Courses: "Udemy: https://www.udemy.com/courses/search/?q=[skill_name]"
  * edX Free Courses: "edX: https://www.edx.org/search?q=[skill_name]"
  * Medium Articles: "Medium: https://medium.com/search?q=[skill_name]"
- Be specific to Indian job market realities
- estimated_weeks per step should be realistic (2-8 weeks each)
- already_have should only include skills from the current_skills list that are relevant to target role
- missing should be skills needed for target role that are NOT in current_skills"""

    try:
        client = groq_sdk.Groq(
            api_key=current_app.config["GROQ_API_KEY"],
            timeout=60.0
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a career advisor. Always respond with valid JSON only. No markdown, no backticks, no explanation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=2048,
            temperature=0.7,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown backticks if model adds them anyway
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)
        return {"success": True, "data": result}

    except json.JSONDecodeError as e:
        logger.error("[CareerService] JSON parse failed: %s", e)
        return {"success": False, "error": "AI returned malformed response. Please try again."}

    except Exception as e:
        logger.error("[CareerService] Career path generation failed: %s", e)
        return {"success": False, "error": "Failed to generate career path. Please try again."}


def generate_skill_gap(user_skills: list, target_job_title: str) -> dict:
    """
    Queries jobs DB for top 20 matching jobs,
    extracts most common skills, compares against user skills,
    returns gap analysis via DeepSeek V3.
    """
    from collections import Counter
    from app.models.job import Job
    import openai

    # ── Step 1: Query DB for matching jobs ───────────────────────────────────
    try:
        matching_jobs = Job.query.filter(
            Job.title.ilike(f'%{target_job_title}%'),
            Job.is_active == True
        ).order_by(Job.posted_at.desc()).limit(20).all()
    except Exception as e:
        logger.error("[SkillGap] DB query failed: %s", e)
        return {"success": False, "error": "Failed to query jobs database."}

    # ── Step 2: Extract and count skills from job listings ───────────────────
    all_skills = []
    for job in matching_jobs:
        if job.skills and isinstance(job.skills, list):
            all_skills.extend([s.lower().strip() for s in job.skills if s])

    if not all_skills:
        # No jobs found — fall back to pure AI analysis
        top_skills = []
        jobs_found = 0
    else:
        skill_counts = Counter(all_skills)
        top_skills = [skill for skill, _ in skill_counts.most_common(15)]
        jobs_found = len(matching_jobs)

    # ── Step 3: Call DeepSeek for intelligent gap analysis ───────────────────
    user_skills_str = ", ".join(user_skills) if user_skills else "None"
    top_skills_str  = ", ".join(top_skills) if top_skills else "Not available from DB"

    prompt = f"""You are a career advisor for the Indian job market.

A professional wants to become a: {target_job_title}
Their current skills: {user_skills_str}
Top skills required by real job listings for this role: {top_skills_str}

Analyze the skill gap and respond ONLY with valid JSON. No preamble, no markdown, no backticks.

{{
  "matching_skills": ["skills user has that are relevant to target role"],
  "missing_skills": ["important skills for target role that user lacks"],
  "match_percentage": 65,
  "priority_skills": ["top 3 most important missing skills to learn first"],
  "recommended_courses": [
    {{
      "skill": "skill name",
      "resource": "specific course or resource name",
      "url": "actual URL",
      "is_free": true
    }}
  ],
  "market_insight": "2 sentences about demand for this role in Indian job market right now"
}}

Rules:
- match_percentage must be a realistic integer 0-100
- recommended_courses must have one entry per priority skill
- For 'url', do NOT hallucinate direct links (they 404). You MUST use one of these deterministic search formats suitable for ANY domain (Finance, HR, Tech, Design, etc.):
  * "https://www.youtube.com/results?search_query=[skill_name]+course"
  * "https://www.coursera.org/search?query=[skill_name]"
  * "https://www.udemy.com/courses/search/?q=[skill_name]"
  * "https://www.edx.org/search?q=[skill_name]"
  * "https://medium.com/search?q=[skill_name]"
- missing_skills should be ordered by importance"""

    try:
        client = openai.OpenAI(
            api_key=current_app.config["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            timeout=90.0
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are a career advisor. Always respond with valid JSON only. No markdown, no backticks, no explanation."
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

        # Strip markdown backticks if model adds them anyway
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)
        result["jobs_analyzed"] = jobs_found
        return {"success": True, "data": result}

    except json.JSONDecodeError as e:
        logger.error("[SkillGap] JSON parse failed: %s", e)
        return {"success": False, "error": "AI returned malformed response. Please try again."}

    except Exception as e:
        logger.error("[SkillGap] Skill gap generation failed: %s", e)
        return {"success": False, "error": "Failed to generate skill gap analysis. Please try again."}


# ── In-memory cache for company research ─────────────────────────────────────
_company_cache: dict = {}


def get_company_research(company_name: str, location: str = "") -> dict:
    """
    Research a company using AI (DeepSeek → Groq fallback).
    Returns rich structured data for the frontend including map_query,
    genuine_reviews, work_environment, hiring_roles, headquarters,
    and platform URLs. Results cached in memory per company name.
    """
    import openai

    cache_key = company_name.lower().strip()

    # Return cached result if available
    if cache_key in _company_cache:
        logger.info("[CompanyResearch] Cache hit for '%s'", company_name)
        return {"success": True, "data": _company_cache[cache_key]}

    location_hint = f" (headquartered in or known presence in {location})" if location else ""

    prompt = f"""You are a senior tech industry researcher with access to Glassdoor, LinkedIn, AmbitionBox, Blind, Levels.fyi, and public company data as of 2025.

Research the company: **{company_name}**{location_hint}

Return ONLY a valid JSON object. No preamble, no markdown, no backticks. Use this exact schema:

{{
  "summary": "<2-3 sentence factual overview: what the company does, market position, notable achievements>",
  "founded": "<year or 'Unknown'>",
  "size": "<headcount range, e.g. '10,000–50,000 employees'>",
  "headquarters": "<full city + country, e.g. 'Bangalore, Karnataka, India' or 'Mountain View, CA, USA'>",
  "map_query": "<city + country optimised for Google Maps, e.g. 'Google HQ Mountain View California'>",
  "website": "<official website URL>",
  "linkedin_url": "<LinkedIn company page URL>",
  "glassdoor_url": "<Glassdoor company page URL>",
  "ambitionbox_url": "<AmbitionBox company page URL if Indian company, else null>",
  "glassdoor_rating": "<e.g. '4.1 / 5' or null>",
  "ambitionbox_rating": "<e.g. '3.9 / 5' or null — only for Indian companies>",
  "tech_stack": ["<tech 1>", "<tech 2>", "..."],
  "culture_notes": "<3-4 sentences on culture: values, pace, management style, team dynamics>",
  "work_environment": {{
    "remote_policy": "<Remote / Hybrid / In-office — with details>",
    "work_hours": "<typical hours/week, e.g. '40-50 hrs/week, flexible'>",
    "perks": ["<perk 1>", "<perk 2>", "<perk 3>", "..."],
    "dress_code": "<Casual / Business casual / Formal>",
    "office_vibe": "<one sentence on physical/virtual workspace feel>"
  }},
  "hiring_roles": {{
    "common_roles": ["<role 1>", "<role 2>", "<role 3>", "..."],
    "top_departments": ["<dept 1>", "<dept 2>", "<dept 3>"],
    "typical_profile": "<2 sentences: what background/skills they typically look for>",
    "seniority_mix": "<e.g. '60% mid-level, 25% senior, 15% fresh'>",
    "preferred_background": ["<college/bootcamp/FAANG exp>", "..."]
  }},
  "hiring_process": "<step-by-step description of typical interview process, 3-5 steps>",
  "interview_tips": ["<tip 1>", "<tip 2>", "<tip 3>", "<tip 4>"],
  "genuine_reviews": [
    {{
      "source": "Glassdoor",
      "role": "<reviewer job title>",
      "rating": <float 1.0-5.0>,
      "sentiment": "positive | mixed | negative",
      "pros": "<what the reviewer liked>",
      "cons": "<what the reviewer disliked>"
    }},
    {{
      "source": "AmbitionBox",
      "role": "<reviewer job title>",
      "rating": <float 1.0-5.0>,
      "sentiment": "positive | mixed | negative",
      "pros": "<what the reviewer liked>",
      "cons": "<what the reviewer disliked>"
    }},
    {{
      "source": "Blind",
      "role": "<reviewer job title>",
      "rating": <float 1.0-5.0>,
      "sentiment": "positive | mixed | negative",
      "pros": "<what the reviewer liked>",
      "cons": "<what the reviewer disliked>"
    }}
  ]
}}

Rules:
- Be SPECIFIC and FACTUAL. Use real known data about {company_name}.
- genuine_reviews must feel like real employee reviews, based on what is publicly known about this company.
- Do NOT make up glassdoor_rating — use known approximate values or null.
- tech_stack must be real technologies this company actually uses.
- If company is Indian, include ambitionbox data.
- map_query should be specific enough to pin the right location on Google Maps.
- Return ONLY the JSON object. No commentary."""

    def _call_deepseek(prompt_text: str) -> str:
        client = openai.OpenAI(
            api_key=current_app.config["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            timeout=60.0,
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are a company research expert. Always respond with valid JSON only. No markdown, no backticks."
                },
                {"role": "user", "content": prompt_text},
            ],
            max_tokens=2048,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def _call_groq(prompt_text: str) -> str:
        import groq
        groq_client = groq.Groq(api_key=current_app.config.get("GROQ_API_KEY", ""))
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a company research expert. Always respond with valid JSON only. No markdown, no backticks."
                },
                {"role": "user", "content": prompt_text},
            ],
            max_tokens=2048,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def _clean_json(raw: str) -> dict:
        """Strip markdown fences and parse JSON."""
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    try:
        try:
            raw = _call_deepseek(prompt)
        except Exception as e:
            logger.warning("[CompanyResearch] DeepSeek failed, trying Groq fallback: %s", e)
            raw = _call_groq(prompt)

        data = _clean_json(raw)

        # Normalise: ensure all expected keys exist (graceful degradation)
        data.setdefault("summary", "")
        data.setdefault("founded", "Unknown")
        data.setdefault("size", "")
        data.setdefault("tech_stack", [])
        data.setdefault("culture_notes", "")
        data.setdefault("work_environment", {})
        data.setdefault("hiring_roles", {})
        data.setdefault("hiring_process", "")
        data.setdefault("interview_tips", [])
        data.setdefault("genuine_reviews", [])
        data.setdefault("glassdoor_rating", None)
        data.setdefault("ambitionbox_rating", None)
        data.setdefault("headquarters", "")
        data.setdefault("map_query", company_name)
        data.setdefault("website", None)
        data.setdefault("linkedin_url", None)
        data.setdefault("glassdoor_url", None)
        data.setdefault("ambitionbox_url", None)

        # Cache the result
        _company_cache[cache_key] = data
        logger.info("[CompanyResearch] Cached result for '%s'", company_name)

        return {"success": True, "data": data}

    except json.JSONDecodeError as e:
        logger.error("[CompanyResearch] JSON parse failed: %s", e)
        return {"success": False, "error": "AI returned malformed response. Please try again."}

    except Exception as e:
        logger.error("[CompanyResearch] Failed for '%s': %s", company_name, e)
        return {"success": False, "error": "Failed to fetch company research. Please try again."}