import logging
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
    import json

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
- Resources must be real: Coursera, YouTube, official docs, freeCodeCamp, Kaggle, etc.
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
    import json
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
    user_skills_str   = ", ".join(user_skills) if user_skills else "None"
    top_skills_str    = ", ".join(top_skills) if top_skills else "Not available from DB"

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
- Use only real, working URLs (Coursera, YouTube, Kaggle, official docs)
- missing_skills should be ordered by importance"""

    try:
        client = openai.OpenAI(
            api_key=current_app.config["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
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


def get_company_research(company_name: str) -> dict:
    """
    Returns research on a company using DeepSeek V3.
    Results cached in memory per company name to avoid repeated API calls.
    """
    import json
    import openai

    cache_key = company_name.lower().strip()

    # Return cached result if available
    if cache_key in _company_cache:
        logger.info("[CompanyResearch] Cache hit for '%s'", company_name)
        return {"success": True, "data": _company_cache[cache_key]}

    prompt = f"""You are a company research expert with deep knowledge of the Indian and global tech job market.

Research this company for a job seeker: {company_name}

Respond ONLY with valid JSON. No preamble, no markdown, no backticks.

{{
  "summary": "2-3 sentence overview of what the company does",
  "industry": "primary industry",
  "founded": "year founded",
  "headquarters": "city, country",
  "india_presence": "description of their India offices/teams or 'No India presence'",
  "tech_stack": ["technology 1", "technology 2", "technology 3"],
  "culture_notes": "2 sentences about work culture, values, work-life balance",
  "hiring_process": "typical hiring process steps for this company",
  "interview_tips": [
    "specific tip 1 for interviewing at this company",
    "specific tip 2",
    "specific tip 3"
  ],
  "common_interview_questions": [
    "example question they commonly ask",
    "example question 2"
  ],
  "glassdoor_rating": 4.1,
  "avg_salary_india_lpa": "range like 12-25 LPA or 'Not available'",
  "pros": ["pro 1", "pro 2", "pro 3"],
  "cons": ["con 1", "con 2"]
}}

Rules:
- glassdoor_rating must be a realistic float between 1.0 and 5.0, or null if unknown
- tech_stack must be real technologies this company actually uses
- interview_tips must be specific to this company, not generic advice
- If company is unknown or very obscure, still return best-effort data with honest uncertainty"""

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
                    "content": "You are a company research expert. Always respond with valid JSON only. No markdown, no backticks."
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

        # Cache the result
        _company_cache[cache_key] = result
        logger.info("[CompanyResearch] Cached result for '%s'", company_name)

        return {"success": True, "data": result}

    except json.JSONDecodeError as e:
        logger.error("[CompanyResearch] JSON parse failed: %s", e)
        return {"success": False, "error": "AI returned malformed response. Please try again."}

    except Exception as e:
        logger.error("[CompanyResearch] Failed for '%s': %s", company_name, e)
        return {"success": False, "error": "Failed to fetch company research. Please try again."}