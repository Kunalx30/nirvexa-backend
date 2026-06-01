"""
app/services/resume_builder_service.py
NyrVexa — Resume Builder: AI generation + LaTeX injection + compilation.

PUBLIC API (called by app/routes/resume.py):
  build_resume(form_data, template_id, user_id, resume_id) -> dict
  enhance_bullets(bullets) -> list

INTERNAL FLOW of build_resume:
  1. Call Groq LLaMA 3.3 70B → generate full resume content as JSON
  2. If word_count < 400 → retry with expansion instruction (max 1 retry)
  3. inject_placeholders() → fill {{PLACEHOLDERS}} in template .tex
  4. compile_latex() → tectonic → PDF bytes
  5. upload_to_r2() → Cloudflare R2 (skipped in local dev if R2 not configured)
  6. Return dict with pdf_bytes, pdf_url, ai_content, latex_code

BLOCK ROUTING uses template.slug (the metadata.json "id" string) NOT the UUID.

TEMPLATE SLUG → LAYOUT → BUILDER MAPPING:
  template_01_modern_blue  → dark navy mdframed header, paracol body     → _paracol builders
  template_02_teal_clean   → dark sidebar 32% left / white panel 68%     → _sidebar builders (skills/edu/certs)
                                                                          → _paracol builders (exp/projects)
  template_03_classic_black→ centered scshape, tabularx/resumeSubheading → _classic builders
  template_04_bold_header  → charcoal+gold mdframed header, paracol body → _paracol builders
  template_05_two_column   → colored header, minipage LEFT 61%/RIGHT 36% → _twocol builders
  template_06_minimal_mono → tabular name+contact header, 58/42 paracol  → minipage-safe builders
"""

import os
import re
import json
import logging
import requests as req

from app.models.resume_template import ResumeTemplate
from app.services.latex_compiler import compile_latex

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
GROQ_RESUME_KEY = lambda: os.environ.get("GROQ_RESUME_API_KEY", "")
GROQ_MAIN_KEY   = lambda: os.environ.get("GROQ_API_KEY", "")
DEEPSEEK_KEY    = lambda: os.environ.get("DEEPSEEK_API_KEY", "")
GEMINI_KEY      = lambda: os.environ.get("GEMINI_API_KEY", "")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LaTeX escaping
# ─────────────────────────────────────────────────────────────────────────────

def latex_escape(text: str) -> str:
    """
    Escape LaTeX special characters that appear in user/AI text.
    We deliberately do NOT escape backslash, { or } here because:
    - User/AI text never legitimately contains raw LaTeX commands
    - Escaping { and } causes double-escaping of our own \& \% etc sequences
    The only chars that realistically appear in names, titles, bullets:
    & % $ # _ ^ ~ < >
    """
    if not text:
        return ""
    text = str(text)
    text = (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace("×", "x")
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("‘", "'")
        .replace(" ? ", " - ")
    )
    chars = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "^": r"\^{}",
        "~": r"\textasciitilde{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    return "".join(chars.get(c, c) for c in text)


def le(text) -> str:
    """Short alias for latex_escape."""
    return latex_escape(str(text)) if text else ""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Groq AI generation
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert ATS resume writer. You write resumes for ANY profession and domain.
You produce clean, professional, ONE-PAGE resumes that look like real human-written resumes.
The resume MUST fit on exactly one A4 page — no overflow, no blank bottom half.

CONTENT TARGETS — follow exactly to hit one full page:
- Summary: exactly 3 sentences. Each sentence 20-28 words. Specific, not generic.
- Experience: exactly 3 bullets per role. Each bullet ONE sentence, 20-28 words.
- Projects: exactly 2 bullets per project. Each bullet ONE sentence, 20-25 words.
- Skills: exactly 5 categories, 4-5 skills each. Use ALL user-provided skills, no duplicates.
- Education: degree + institution + CGPA + exactly 5 coursework subjects (comma-separated string, NOT a list).
- Certifications: full name + issuer + year.

BULLET QUALITY — every bullet MUST follow this formula:
[Strong Action Verb] + [Specific technical task with tools/methods] + [Quantified result with a number]
- REJECTED (too short/vague): "Built ML models." / "Analyzed data." / "Developed pipelines."
- REJECTED (no number): "Improved prediction accuracy significantly using Scikit-learn models."
- GOOD: "Developed and deployed 3 Scikit-learn classification models on client datasets, improving prediction accuracy by 28% over the existing rule-based baseline."
- GOOD: "Designed automated Python and SQL ETL pipelines processing 500K+ daily records, reducing manual data preparation effort by 40% across 3 teams."
- GOOD: "Built interactive Power BI dashboards tracking 5 business KPIs, enabling stakeholders to reduce weekly reporting time from 3 days to 4 hours."
- If user gave no metrics, invent realistic industry-standard numbers. Always include at least ONE specific number.

SUMMARY QUALITY:
- Sentence 1: role title + top 3 specific skills/tools.
- Sentence 2: current experience/internship with what they do and a measurable outcome.
- Sentence 3: value they bring + career goal.
- BAD: "Data Scientist with Python skills. Experienced. Bringing insights." — too short, rejected.
- GOOD: "B.Tech CSE graduate with hands-on experience in ML pipelines, Flask deployment, and Power BI dashboards."

MINIMAL INPUT HANDLING:
- If bullets_raw is empty for a role: invent 3 strong realistic bullets using only role title, company, and tech skills.
- If project description is empty: invent 2 strong impact-focused bullets using project name and tech stack.
- NEVER leave any section empty. Always generate meaningful professional content.

SKILLS CATEGORY NAMES — domain-appropriate, no ampersands:
- Data/ML: Languages and Libraries, ML and Deep Learning, Deployment and Backend, Data and BI, Tools
- Software: Languages, Frameworks, Tools, Databases, Cloud
- Marketing: Platforms, Analytics Tools, Design Tools, Skills
- Other domains: adapt naturally to the profession.

Return ONLY valid JSON. No markdown. No explanation. No preamble."""


def _build_user_prompt(form_data: dict) -> str:
    name        = form_data.get("full_name", "")
    email       = form_data.get("email", "")
    phone       = form_data.get("phone", "")
    location    = form_data.get("location", "")
    linkedin    = form_data.get("linkedin_url", "")
    github      = form_data.get("github_url", "")
    target_role = form_data.get("target_role", "Professional")
    jd_text     = form_data.get("job_description", "").strip()
    education   = form_data.get("education", [])
    experience  = form_data.get("experience", [])
    projects    = form_data.get("projects", [])
    skills_raw  = form_data.get("skills_raw", "")
    certs_raw   = form_data.get("certifications_raw", "")

    jd_section = f"\nJOB DESCRIPTION (optimize keywords for this):\n{jd_text}\n" if jd_text else ""

    edu_text = "\n".join(
        f"  - {e.get('degree','')} from {e.get('institution','')} ({e.get('year','')}), CGPA/Percentage: {e.get('cgpa','')}"
        for e in (education if isinstance(education, list) else [])
    )
    exp_text = "\n".join(
        f"  - {e.get('role','')} at {e.get('company','')} ({e.get('duration','')})\n"
        f"    Raw bullets: {e.get('bullets_raw', e.get('bullets', ''))}"
        for e in (experience if isinstance(experience, list) else [])
    )
    proj_text = "\n".join(
        f"  - {p.get('name','')} | Tech: {p.get('tech','')} | Desc: {p.get('description','')}"
        for p in (projects if isinstance(projects, list) else [])
    )

    schema = '''{
  "summary": "B.Tech CSE graduate with hands-on experience building ML pipelines using Python, Scikit-learn, and Flask. Currently interning at InfozIT Solutions applying supervised learning and data pipeline engineering on real client datasets. Bringing strong foundations in feature engineering, model deployment, and data visualization to drive measurable business outcomes.",
  "experience": [
    {
      "company": "Company Name",
      "role": "Job Title",
      "duration": "Mar 2026 – Present",
      "location": "City, State",
      "bullets": [
        "Developed and deployed 3 Scikit-learn classification models on client datasets, improving prediction accuracy by 28% over the existing rule-based baseline system.",
        "Designed automated Python and SQL ETL pipelines processing 500K+ daily records, reducing manual data preparation effort by 40% across 3 business teams.",
        "Built interactive Power BI dashboards tracking 5 key business KPIs, enabling stakeholders to cut weekly reporting time from 3 days to 4 hours."
      ]
    }
  ],
  "projects": [
    {
      "name": "Project Name",
      "tech": "Python, Flask, PostgreSQL, Groq API",
      "description": [
        "Built an AI-powered job platform matching candidates to openings using NLP-based semantic search and personalized recommendations, serving 1000+ active users.",
        "Deployed on Flask and PostgreSQL with a FAISS similarity engine and Groq API integration, achieving 50% increase in user engagement and 35% faster job matching."
      ],
      "github_url": "",
      "live_url": ""
    }
  ],
  "skills": {
    "Languages and Libraries": ["Python", "SQL", "Pandas", "NumPy"],
    "ML and Deep Learning": ["Scikit-learn", "XGBoost", "Random Forest", "Ridge Regression"],
    "Deployment and Backend": ["Flask", "FastAPI", "Streamlit", "REST APIs"],
    "Data and BI": ["Power BI", "Matplotlib", "Seaborn", "EDA", "Feature Engineering"],
    "Tools": ["Git", "Jupyter", "VS Code", "Statistics", "Linear Algebra"]
  },
  "education": [
    {
      "degree": "Bachelor of Technology in Computer Science and Engineering",
      "institution": "RSR Rungta College of Engineering and Technology, Bhilai",
      "year": "2021 – 2025",
      "cgpa": "7.3 / 10",
      "coursework": "Machine Learning, Database Management Systems, Data Structures and Algorithms, Statistics, Operating Systems"
    }
  ],
  "certifications": [
    {"name": "Post Graduate Certification in Data Science", "issuer": "E and ICT Academy IIT Guwahati", "date": "2026"}
  ],
  "word_count": 480
}'''

    return f"""Generate a complete ATS-optimized resume for this person.
Target role: {target_role}
Adapt all skill category names and content language to match this domain.

PERSONAL INFO:
Name: {name} | Email: {email} | Phone: {phone} | Location: {location}
LinkedIn: {linkedin} | GitHub: {github}
{jd_section}
EDUCATION:
{edu_text if edu_text else "  Not provided"}

WORK EXPERIENCE:
{exp_text if exp_text else "  No experience provided — write strong project bullets and expand education section"}

PROJECTS:
{proj_text if proj_text else "  Not provided"}

SKILLS (raw input — categorize all of these into domain-appropriate categories): {skills_raw}
CERTIFICATIONS: {certs_raw}

CRITICAL — YOUR OUTPUT WILL BE REJECTED IF:
- Any bullet is under 12 words (e.g. "Built ML models." = REJECTED)
- Summary is under 3 full sentences
- Skills has fewer than 5 categories
- Any skill category has fewer than 3 skills
- Any project has fewer than 3 bullet points

Every bullet MUST have: action verb + specific technical detail + number/metric.
Skill category names MUST match the domain of "{target_role}".
Use ALL skills from the raw input — never drop any skill.
No skill appears in more than one category.

Use THIS exact JSON structure — replace ALL example values with real content for this person:
{schema}"""


def _call_ai_generate(form_data: dict, extra_instruction: str = "") -> dict:
    """
    Generate resume content JSON using a multi-provider fallback chain.
    Order: Groq (resume key) → Groq (main key) → DeepSeek → Gemini.
    """
    system = SYSTEM_PROMPT
    if extra_instruction:
        system += f"\n\nADDITIONAL INSTRUCTION: {extra_instruction}"

    user_prompt = _build_user_prompt(form_data)
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_prompt},
    ]

    # ── Provider 1: Groq with RESUME API key ──────────────────────
    groq_resume_key = GROQ_RESUME_KEY()
    if groq_resume_key:
        try:
            logger.info("[Resume AI] Trying Groq (resume key)")
            raw = _call_openai_compat(GROQ_URL, groq_resume_key, "llama-3.3-70b-versatile", messages)
            return _parse_ai_response(raw)
        except Exception as e:
            logger.warning("[Resume AI] Groq resume key failed: %s", e)

    # ── Provider 2: Groq with MAIN API key ────────────────────────
    groq_main_key = GROQ_MAIN_KEY()
    if groq_main_key and groq_main_key != groq_resume_key:
        try:
            logger.info("[Resume AI] Trying Groq (main key)")
            raw = _call_openai_compat(GROQ_URL, groq_main_key, "llama-3.3-70b-versatile", messages)
            return _parse_ai_response(raw)
        except Exception as e:
            logger.warning("[Resume AI] Groq main key failed: %s", e)

    # ── Provider 3: DeepSeek ──────────────────────────────────────
    ds_key = DEEPSEEK_KEY()
    if ds_key:
        try:
            logger.info("[Resume AI] Trying DeepSeek")
            raw = _call_openai_compat(DEEPSEEK_URL, ds_key, "deepseek-chat", messages)
            return _parse_ai_response(raw)
        except Exception as e:
            logger.warning("[Resume AI] DeepSeek failed: %s", e)

    # ── Provider 4: Gemini ────────────────────────────────────────
    gemini_key = GEMINI_KEY()
    if gemini_key:
        try:
            logger.info("[Resume AI] Trying Gemini")
            raw = _call_gemini_resume(gemini_key, system, user_prompt)
            return _parse_ai_response(raw)
        except Exception as e:
            logger.warning("[Resume AI] Gemini failed: %s", e)

    raise RuntimeError("[Resume AI] All providers failed. Cannot generate resume content.")


def _call_openai_compat(url: str, api_key: str, model: str, messages: list) -> str:
    """Call any OpenAI-compatible API (Groq, DeepSeek) via raw HTTP."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.65,
        "max_tokens": 2800,
    }
    resp = req.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_gemini_resume(api_key: str, system: str, user_prompt: str) -> str:
    """Call Google Gemini for resume generation."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system,
        generation_config=genai.GenerationConfig(
            max_output_tokens=2800,
            temperature=0.65,
        ),
    )
    response = model.generate_content(user_prompt, request_options={"timeout": 90})
    return response.text


def _parse_ai_response(raw: str) -> dict:
    """Parse and sanitize the AI-generated JSON resume content."""
    clean = re.sub(r"```json|```", "", raw).strip()
    parsed = json.loads(clean)
    parsed = _sanitize_ampersands(parsed)
    parsed = _deduplicate_skills(parsed)
    return parsed


def _sanitize_ampersands(obj):
    """
    Recursively walk AI JSON output and replace bare & with 'and'.
    We do NOT pre-escape to \\& here because latex_escape() will handle
    escaping at render time. Pre-escaping causes double-escape.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_ampersands(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_ampersands(v) for v in obj]
    elif isinstance(obj, str):
        return obj.replace("&", "and")
    return obj


def _format_category_label(raw: str) -> str:
    """
    Convert AI-returned category key to a readable label.
    Keeps 'and', 'or', 'the' lowercase (prepositions/conjunctions).
    Uppercases known acronyms: ML, BI, NLP, AI, SQL, etc.
    """
    ACRONYMS = {
        "ml", "bi", "nlp", "ai", "sql", "css", "html", "js", "api",
        "ux", "ui", "hr", "erp", "crm", "seo", "sem", "etl", "ci",
        "cd", "aws", "gcp", "sdk", "orm", "ide", "os", "db",
    }
    LOWERCASE_WORDS = {"and", "or", "the", "of", "in", "for", "with", "a", "an"}
    words = raw.replace("_", " ").replace("-", " ").split()
    result = []
    for i, word in enumerate(words):
        lower = word.lower()
        if lower in ACRONYMS:
            result.append(word.upper())
        elif lower in LOWERCASE_WORDS and i != 0:
            result.append(lower)
        else:
            result.append(word.capitalize())
    return " ".join(result)


def _normalize_coursework(value) -> str:
    """
    Coursework may come back as a Python list or a comma-separated string.
    Always return a clean, LaTeX-escaped comma-separated string.
    """
    if isinstance(value, list):
        return ", ".join(le(str(v).strip()) for v in value)
    return le(str(value).strip())


def _deduplicate_skills(obj: dict) -> dict:
    """
    Remove duplicate skills that appear in multiple categories.
    First occurrence wins — preserves the most specific category placement.
    """
    skills = obj.get("skills", {})
    if not isinstance(skills, dict):
        return obj
    seen = set()
    cleaned = {}
    for category, items in skills.items():
        if not isinstance(items, list):
            continue
        deduped = []
        for skill in items:
            key = str(skill).strip().lower()
            if key not in seen:
                seen.add(key)
                deduped.append(skill)
        if deduped:
            cleaned[category] = deduped
    obj["skills"] = cleaned
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Block builders: Paracol layout
# template_01_modern_blue, template_04_bold_header, template_06_minimal_mono
# Also used for experience/projects in template_02_teal_clean right panel
# ─────────────────────────────────────────────────────────────────────────────

def _skills_paracol(skills: dict) -> str:
    lines = []
    for category_name, items in skills.items():
        if not items:
            continue
        label = _format_category_label(category_name)
        escaped_label = le(label)
        escaped_items = ", ".join(le(str(i)) for i in items)
        lines.append(f"\\textbf{{{escaped_label}:}} {escaped_items}\\\\[2pt]")
    return "\n".join(lines)


def _education_paracol(education: list) -> str:
    blocks = []
    for edu in education:
        degree      = le(edu.get("degree", ""))
        institution = le(edu.get("institution", ""))
        year        = le(edu.get("year", ""))
        cgpa        = le(edu.get("cgpa", ""))
        coursework  = _normalize_coursework(edu.get("coursework", ""))

        block = (
            f"    \\begin{{twocolentry}}{{\\textbf{{{year}}}}}\n"
            f"        \\textbf{{{institution}}}\n"
            f"    \\end{{twocolentry}}\n"
            f"    \\vspace{{0.01cm}}\n"
            f"    \\begin{{onecolentry}}\n"
            f"        \\textit{{{degree}}}"
        )
        if cgpa:
            block += f" \\hfill {{\\small\\textbf{{CGPA: {cgpa}}}}}"
        block += "\n"
        if coursework:
            block += f"        \\\\[2pt]{{\\small Coursework: {coursework}}}\n"
        block += (
            f"    \\end{{onecolentry}}\n"
            f"    \\vspace{{0.04cm}}"
        )
        blocks.append(block)
    return "\n".join(blocks)


def _experience_paracol(experience: list) -> str:
    blocks = []
    for exp in experience:
        company  = le(exp.get("company", ""))
        role     = le(exp.get("role", ""))
        duration = le(exp.get("duration", ""))
        location = le(exp.get("location", ""))
        bullets  = exp.get("bullets", [])

        loc_str = f", {location}" if location else ""
        block = (
            f"    \\begin{{twocolentry}}{{\\textbf{{{duration}}}}}\n"
            f"        \\textbf{{{role}}} $|$ \\textit{{{company}{loc_str}}}\n"
            f"    \\end{{twocolentry}}\n"
            f"    \\vspace{{0.01cm}}\n"
            f"    \\begin{{onecolentry}}\n"
            f"        \\begin{{highlights}}\n"
        )
        for bullet in bullets:
            block += f"            \\item {le(bullet)}\n"
        block += (
            f"        \\end{{highlights}}\n"
            f"    \\end{{onecolentry}}\n"
            f"    \\vspace{{0.04cm}}"
        )
        blocks.append(block)
    return "\n".join(blocks)


def _projects_paracol(projects: list) -> str:
    blocks = []
    for proj in projects:
        name       = le(proj.get("name", ""))
        tech       = le(proj.get("tech", ""))
        desc       = proj.get("description", "")
        live_url   = proj.get("live_url", "")
        github_url = proj.get("github_url", "")

        links = []
        if live_url:
            links.append(f"\\hrefWithoutArrow{{{live_url}}}{{Live}}")
        if github_url:
            links.append(f"\\hrefWithoutArrow{{{github_url}}}{{GitHub}}")
        right_col = " $|$ ".join(links) if links else ""

        desc_items = desc if isinstance(desc, list) else [desc]

        block = (
            f"    \\begin{{twocolentry}}{{{right_col}}}\n"
            f"        \\textbf{{{name}}} $|$ \\textit{{{tech}}}\n"
            f"    \\end{{twocolentry}}\n"
            f"    \\vspace{{0.01cm}}\n"
            f"    \\begin{{onecolentry}}\n"
            f"        \\begin{{highlights}}\n"
        )
        for item in desc_items:
            block += f"            \\item {le(item)}\n"
        block += (
            f"        \\end{{highlights}}\n"
            f"    \\end{{onecolentry}}\n"
            f"    \\vspace{{0.04cm}}"
        )
        blocks.append(block)
    return "\n".join(blocks)


def _certifications_paracol(certifications: list) -> str:
    lines = []
    for cert in certifications:
        name   = le(cert.get("name", ""))
        issuer = le(cert.get("issuer", ""))
        date   = le(cert.get("date", ""))
        line   = f"\\item \\textbf{{{name}}}"
        if issuer:
            line += f" -- {issuer}"
        if date:
            line += f" \\textit{{({date})}}"
        lines.append(f"            {line}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3b — Block builders: Sidebar layout (template_02_teal_clean)
#
# The sidebar is a dark mdframed panel. Skills, education, and certifications
# go here in compact white-text format. Experience and projects go into the
# right white panel using _paracol builders (onecolentry/twocolentry are
# defined in the template without the 1.6cm adjustwidth padding).
# ─────────────────────────────────────────────────────────────────────────────

def _skills_sidebar(skills: dict) -> str:
    """
    Skills block for the dark sidebar.
    Each category: bold teal label on one line, comma-list below.
    """
    lines = []
    for category_name, items in skills.items():
        if not items:
            continue
        label = _format_category_label(category_name)
        escaped_items = ", ".join(le(str(i)) for i in items)
        lines.append(
            f"{{\\bfseries\\color{{sidebarAccent}}\\small {le(label)}:}}\\\\\n"
            f"\\small {escaped_items}\\\\[5pt]\n"
        )
    return "\n".join(lines)


def _education_sidebar(education: list) -> str:
    """
    Education block for the dark sidebar — compact, no twocolentry needed.
    """
    blocks = []
    for edu in education:
        degree      = le(edu.get("degree", ""))
        institution = le(edu.get("institution", ""))
        year        = le(edu.get("year", ""))
        cgpa        = le(edu.get("cgpa", ""))
        coursework  = _normalize_coursework(edu.get("coursework", ""))

        block = (
            f"{{\\bfseries\\small {institution}}}\\\\\n"
            f"{{\\small\\textit{{{degree}}}}}\\\\\n"
            f"{{\\small\\textbf{{{year}}}}}"
        )
        if cgpa:
            block += f"\\\\\n{{\\small\\textbf{{CGPA: {cgpa}}}}}"
        if coursework:
            block += f"\\\\\n{{\\footnotesize Coursework: {coursework}}}"
        block += "\\\\[5pt]"
        blocks.append(block)
    return "\n".join(blocks)


def _experience_minipage(experience: list) -> str:
    """
    Experience for the RIGHT panel of the sidebar template.
    NO paracol/twocolentry — those crash inside minipage.
    Uses simple \hfill tabular-style header line instead.
    """
    blocks = []
    for exp in experience:
        company  = le(exp.get("company", ""))
        role     = le(exp.get("role", ""))
        duration = le(exp.get("duration", ""))
        location = le(exp.get("location", ""))
        bullets  = exp.get("bullets", [])

        loc_str = f", {location}" if location else ""
        block = (
            f"\\noindent\\textbf{{{role}}} $|$ \\textit{{{company}{loc_str}}}"
            f"\\hfill {{\\small\\textbf{{{duration}}}}}\\\\\n"
            f"\\vspace{{0.01cm}}\n"
            f"\\begin{{highlights}}\n"
        )
        for bullet in bullets:
            block += f"    \\item {le(bullet)}\n"
        block += "\\end{highlights}\n\\vspace{0.06cm}\n"
        blocks.append(block)
    return "\n".join(blocks)


def _projects_minipage(projects: list) -> str:
    """
    Projects for the RIGHT panel of the sidebar template.
    NO paracol/twocolentry — those crash inside minipage.
    """
    blocks = []
    for proj in projects:
        name       = le(proj.get("name", ""))
        tech       = le(proj.get("tech", ""))
        desc       = proj.get("description", "")
        github_url = proj.get("github_url", "")
        live_url   = proj.get("live_url", "")

        desc_items = desc if isinstance(desc, list) else [desc]
        links = []
        if github_url:
            links.append(f"\\href{{{github_url}}}{{GitHub}}")
        if live_url:
            links.append(f"\\href{{{live_url}}}{{Live}}")
        link_str = " $|$ ".join(links)
        link_str = f" \\hfill {{\\small {link_str}}}" if link_str else ""

        block = (
            f"\\noindent\\textbf{{{name}}}{link_str}\\\\\n"
            f"{{\\small\\textit{{{tech}}}}}\\\\\n"
            f"\\vspace{{0.01cm}}\n"
            f"\\begin{{highlights}}\n"
        )
        for item in desc_items:
            block += f"    \\item {le(item)}\n"
        block += "\\end{highlights}\n\\vspace{0.06cm}\n"
        blocks.append(block)
    return "\n".join(blocks)


def _certifications_sidebar(certifications: list) -> str:
    """
    Certifications block for the dark sidebar.
    Rendered inside highlightsforbulletentries (teal bullet, white text).
    """
    lines = []
    for cert in certifications:
        name   = le(cert.get("name", ""))
        issuer = le(cert.get("issuer", ""))
        date   = le(cert.get("date", ""))
        line   = f"\\item {{\\small\\textbf{{{name}}}}}"
        if issuer:
            line += f"\\\\ {{\\footnotesize\\textit{{{issuer}}}}}"
        if date:
            line += f" {{\\footnotesize ({date})}}"
        lines.append(line)
    return "\n        ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Block builders: Classic Black (template_03_classic_black)
# ─────────────────────────────────────────────────────────────────────────────

def _skills_classic(skills: dict) -> str:
    lines = []
    for category_name, items in skills.items():
        if not items:
            continue
        label = _format_category_label(category_name)
        escaped_label = le(label)
        escaped_items = ", ".join(le(str(i)) for i in items)
        lines.append(f"        \\textbf{{{escaped_label}}}{{: {escaped_items}}} \\\\")
    return "\n".join(lines)


def _education_classic(education: list) -> str:
    blocks = []
    for edu in education:
        degree      = le(edu.get("degree", ""))
        institution = le(edu.get("institution", ""))
        year        = le(edu.get("year", ""))
        cgpa        = le(edu.get("cgpa", ""))
        blocks.append(
            f"    \\resumeSubheading\n"
            f"      {{{institution}}}{{}}\n"
            f"      {{{degree}}}{{\\textbf{{{year}}}}}\n"
        )
        if cgpa:
            blocks.append(
                f"    \\resumeSubSubheading\n"
                f"     {{\\textbf{{CGPA/Percentage: {cgpa}}}}}{{}}\n"
            )
    return "\n".join(blocks)


def _experience_classic(experience: list) -> str:
    blocks = []
    for exp in experience:
        company  = le(exp.get("company", ""))
        role     = le(exp.get("role", ""))
        duration = le(exp.get("duration", ""))
        location = le(exp.get("location", ""))
        bullets  = exp.get("bullets", [])

        block = (
            f"    \\resumeSubheading\n"
            f"      {{{role}}}{{\\textbf{{{duration}}}}}\n"
            f"      {{{company}}}{{{location}}}\n"
            f"      \\resumeItemListStart\n"
        )
        for bullet in bullets:
            block += f"        \\resumeItem{{{le(bullet)}}}\n"
        block += "      \\resumeItemListEnd\n"
        blocks.append(block)
    return "\n".join(blocks)


def _projects_classic(projects: list) -> str:
    blocks = []
    for proj in projects:
        name       = le(proj.get("name", ""))
        tech       = le(proj.get("tech", ""))
        desc       = proj.get("description", "")
        github_url = proj.get("github_url", "")

        right = f"\\href{{{github_url}}}{{GitHub}}" if github_url else ""
        desc_items = desc if isinstance(desc, list) else [desc]

        block = (
            f"      \\resumeProjectHeading\n"
            f"          {{\\textbf{{{name}}} $|$ \\emph{{{tech}}}}}{{{right}}}\n"
            f"          \\resumeItemListStart\n"
        )
        for item in desc_items:
            block += f"            \\resumeItem{{{le(item)}}}\n"
        block += "          \\resumeItemListEnd\n"
        blocks.append(block)
    return "\n".join(blocks)


def _certifications_classic(certifications: list) -> str:
    lines = []
    for cert in certifications:
        name   = le(cert.get("name", ""))
        issuer = le(cert.get("issuer", ""))
        date   = le(cert.get("date", ""))
        line   = f"\\textbf{{{name}}}"
        if issuer:
            line += f" -- {issuer}"
        if date:
            line += f" ({date})"
        lines.append(line)
    return " \\\\\n        ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Block builders: Two Column (template_05_two_column)
# ─────────────────────────────────────────────────────────────────────────────

def _experience_twocol(experience: list) -> str:
    blocks = []
    for exp in experience:
        company  = le(exp.get("company", ""))
        role     = le(exp.get("role", ""))
        duration = le(exp.get("duration", ""))
        location = le(exp.get("location", ""))
        bullets  = exp.get("bullets", [])

        loc_str = f" -- {location}" if location else ""
        block = (
            f"\\textbf{{{role}}} \\hfill {{\\small\\textbf{{{duration}}}}}\\\\\n"
            f"\\textit{{{company}}}{loc_str}\n"
            f"\\begin{{tightitemize}}\n"
        )
        for bullet in bullets:
            block += f"    \\item {le(bullet)}\n"
        block += "\\end{tightitemize}\n\\vspace{0.1cm}\n"
        blocks.append(block)
    return "\n".join(blocks)


def _projects_twocol(projects: list) -> str:
    blocks = []
    for proj in projects:
        name       = le(proj.get("name", ""))
        tech       = le(proj.get("tech", ""))
        desc       = proj.get("description", "")
        github_url = proj.get("github_url", "")

        desc_items = desc if isinstance(desc, list) else [desc]
        link_str = (
            f" \\hfill \\href{{{github_url}}}{{\\small GitHub}}"
            if github_url else ""
        )
        block = (
            f"\\textbf{{{name}}}{link_str}\\\\\n"
            f"{{\\small\\textit{{{tech}}}}}\n"
            f"\\begin{{tightitemize}}\n"
        )
        for item in desc_items:
            block += f"    \\item {le(item)}\n"
        block += "\\end{tightitemize}\n\\vspace{0.1cm}\n"
        blocks.append(block)
    return "\n".join(blocks)


def _education_twocol(education: list) -> str:
    blocks = []
    for edu in education:
        degree      = le(edu.get("degree", ""))
        institution = le(edu.get("institution", ""))
        year        = le(edu.get("year", ""))
        cgpa        = le(edu.get("cgpa", ""))
        coursework  = _normalize_coursework(edu.get("coursework", ""))

        block = f"\\textbf{{{institution}}}\\\\\n\\textit{{\\small {degree}}} \\hfill \\textbf{{{year}}}\\\\\n"
        if cgpa:
            block += f"{{\\small\\textbf{{CGPA: {cgpa}}}}}\\\\\n"
        if coursework:
            block += f"{{\\footnotesize Coursework: {coursework}}}\\\\\n"
        block += "\\vspace{0.15cm}\n"
        blocks.append(block)
    return "\n".join(blocks)


def _skills_twocol(skills: dict) -> str:
    lines = []
    for category_name, items in skills.items():
        if not items:
            continue
        label = _format_category_label(category_name)
        escaped_label = le(label)
        escaped_items = ", ".join(le(str(i)) for i in items)
        lines.append(f"\\textbf{{{escaped_label}:}} {escaped_items}\\\\")
    return "\n".join(lines)


def _certifications_twocol(certifications: list) -> str:
    lines = []
    for cert in certifications:
        name   = le(cert.get("name", ""))
        issuer = le(cert.get("issuer", ""))
        date   = le(cert.get("date", ""))
        line   = f"\\item \\textbf{{{name}}}"
        if issuer:
            line += f"\\\\ \\textit{{\\small {issuer}}}"
        if date:
            line += f" ({date})"
        lines.append(line)
    return "\n    ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Portfolio helper + contact line builder
# ─────────────────────────────────────────────────────────────────────────────

def _portfolio_snippet(portfolio_url: str, style: str = "mono") -> str:
    if not portfolio_url:
        return ""
    url     = portfolio_url.strip()
    display = url.replace("https://", "").replace("http://", "").rstrip("/")
    if style == "white":
        # template_01 + template_04: white pipe separator, white text
        return (
            f" \\enspace\\textcolor{{white}}{{$\\cdot$}}\\enspace"
            f"\\hrefWithoutArrow{{{url}}}{{\\textcolor{{white}}{{{display}}}}}"
        )
    elif style == "mono":
        # legacy AND-chain style
        return f"\n        \\kern 5pt\\AND\\kern 5pt\\mbox{{\\hrefWithoutArrow{{{url}}}{{{display}}}}}"
    elif style == "center":
        # template_03_classic_black: plain pipe
        return f" $|$ \\href{{{url}}}{{{display}}}"
    elif style == "center_color":
        # template_05_two_column: colored pipe
        return f" {{\\color{{accentColor}}$|$}} \\hrefWithoutArrow{{{url}}}{{{display}}}"
    elif style == "exec":
        # template_04_bold_header: gold dot separator, white text
        return (
            f" \\enspace\\textcolor{{goldAccent}}{{$\\cdot$}}\\enspace"
            f"\\hrefWithoutArrow{{{url}}}{{\\textcolor{{white}}{{{display}}}}}"
        )
    return ""


def _build_contact_line(form_data: dict) -> str:
    """
    Builds the centered header contact line (previously used by template_02_teal_clean).
    Kept for backwards compatibility — template_02 now uses sidebar, so this
    placeholder will simply be stripped by the final regex cleanup.
    """
    phone         = form_data.get("phone", "")
    phone_raw     = re.sub(r"[^\d+]", "", phone)
    email         = form_data.get("email", "")
    location      = form_data.get("location", "")
    linkedin_url  = form_data.get("linkedin_url", "")
    github_url    = form_data.get("github_url", "")
    portfolio_url = form_data.get("portfolio_url", "")

    parts = []
    if location:
        parts.append(f"\\mbox{{{le(location)}}}")
    if phone:
        parts.append(f"\\mbox{{\\href{{tel:{phone_raw}}}{{{le(phone)}}}}}")
    if email:
        parts.append(f"\\mbox{{\\href{{mailto:{email}}}{{{le(email)}}}}}")
    if linkedin_url:
        parts.append(f"\\mbox{{\\href{{{linkedin_url}}}{{LinkedIn}}}}")
    if github_url:
        parts.append(f"\\mbox{{\\href{{{github_url}}}{{GitHub}}}}")
    if portfolio_url:
        display = portfolio_url.replace("https://", "").replace("http://", "").rstrip("/")
        if len(display) > 25:
            display = display[:25] + "..."
        parts.append(f"\\mbox{{\\href{{{portfolio_url}}}{{{le(display)}}}}}")

    separator = " \\enspace\\textcolor{accentColor}{$|$}\\enspace "
    return separator.join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Placeholder injection
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3c — Block builders: Warm Sidebar (template_11_warm_sidebar)
#
# Left minipage uses section* (no numbering, no paracol).
# Skills: category label + comma list per line.
# Education: compact name/degree/year/cgpa.
# Certifications: \item entries for the \blacksquare list already in template.
# Experience/Projects: use _experience_minipage / _projects_minipage (already defined).
# ─────────────────────────────────────────────────────────────────────────────

def _skills_warm_sidebar(skills: dict) -> str:
    lines = []
    for category_name, items in skills.items():
        if not items:
            continue
        label = _format_category_label(category_name)
        escaped_items = ", ".join(le(str(i)) for i in items)
        lines.append(f"{{\\small\\bfseries\\color{{accent}} {le(label)}:}}\\\\\n{{\\small {escaped_items}}}\\\\[3pt]")
    return "\n".join(lines)


def _education_warm_sidebar(education: list) -> str:
    blocks = []
    for edu in education:
        degree      = le(edu.get("degree", ""))
        institution = le(edu.get("institution", ""))
        year        = le(edu.get("year", ""))
        cgpa        = le(edu.get("cgpa", ""))
        block = f"{{\\small\\bfseries\\color{{accent}} {institution}}}\\\\\n{{\\small\\itshape {degree}}}"
        if year:
            block += f" \\hfill {{\\small\\textbf{{{year}}}}}"
        if cgpa:
            block += f"\\\\\n{{\\small\\textbf{{CGPA: {cgpa}}}}}"
        block += "\\\\[4pt]"
        blocks.append(block)
    return "\n".join(blocks)


def _certifications_warm_sidebar(certifications: list) -> str:
    lines = []
    for cert in certifications:
        name   = le(cert.get("name", ""))
        issuer = le(cert.get("issuer", ""))
        date   = le(cert.get("date", ""))
        line   = f"\\item {{\\small\\textbf{{{name}}}}}"
        if issuer:
            line += f"\\\\ {{\\footnotesize {issuer}}}"
        if date:
            line += f" {{\\footnotesize ({date})}}"
        lines.append(line)
    return "\n    ".join(lines)


def inject_placeholders(template: "ResumeTemplate", ai_content: dict, form_data: dict) -> str:
    tex  = template.latex_code
    slug = template.slug or ""

    # ── Extract all form fields ──────────────────────────────────────
    full_name     = le(form_data.get("full_name", ""))
    phone         = form_data.get("phone", "")
    phone_raw     = re.sub(r"[^\d+]", "", phone)
    email         = form_data.get("email", "")
    location      = form_data.get("location", "")
    linkedin_url  = form_data.get("linkedin_url", "")
    github_url    = form_data.get("github_url", "")
    portfolio_url = form_data.get("portfolio_url", "")
    exp_title     = le(form_data.get("experience_section_title", "Experience"))
    target_role   = le(form_data.get("target_role", ""))

    linkedin_handle = linkedin_url.rstrip("/").split("/")[-1] if linkedin_url else ""
    github_handle   = github_url.rstrip("/").split("/")[-1] if github_url else ""

    portfolio_display = ""
    if portfolio_url:
        portfolio_display = portfolio_url.replace("https://", "").replace("http://", "").rstrip("/")
        if len(portfolio_display) > 28:
            portfolio_display = portfolio_display[:28] + "..."

    # ── Universal placeholders ───────────────────────────────────────
    tex = tex.replace("{{FULL_NAME}}",                full_name)
    tex = tex.replace("{{EXPERIENCE_SECTION_TITLE}}", exp_title)
    tex = tex.replace("{{SUMMARY}}",                  le(ai_content.get("summary", "")))
    tex = tex.replace("{{TARGET_ROLE_HEADER}}",       target_role)

    # ── Individual contact placeholders ─────────────────────────────
    tex = tex.replace("{{PHONE}}",           le(phone))
    tex = tex.replace("{{PHONE_RAW}}",       phone_raw)
    tex = tex.replace("{{EMAIL}}",           le(email))
    tex = tex.replace("{{LOCATION}}",        le(location))
    tex = tex.replace("{{LINKEDIN_URL}}",    linkedin_url)
    tex = tex.replace("{{GITHUB_URL}}",      github_url)
    tex = tex.replace("{{LINKEDIN_HANDLE}}", le(linkedin_handle))
    tex = tex.replace("{{GITHUB_HANDLE}}",   le(github_handle))

    # ── CONTACT_LINE — kept for backwards compat, stripped if unused ─
    tex = tex.replace("{{CONTACT_LINE}}", _build_contact_line(form_data))

    # ── Portfolio variants ───────────────────────────────────────────
    if portfolio_url and portfolio_display:
        pd = le(portfolio_display)

        # template_01_modern_blue + template_04_bold_header: white text in dark header
        p_white = (
            f" \\enspace\\textcolor{{white}}{{$\\cdot$}}\\enspace"
            f"\\hrefWithoutArrow{{{portfolio_url}}}{{\\textcolor{{white}}{{{pd}}}}}"
        )
        # template_02_teal_clean sidebar: white text, line break style
        p_white_sidebar = f"\\\\\n        \\hrefWithoutArrow{{{portfolio_url}}}{{\\textcolor{{white}}{{{pd}}}}}"
        # template_03_classic_black: plain pipe separator
        p_center = f" $|$ \\href{{{portfolio_url}}}{{{pd}}}"
        # template_05_two_column: accentColor pipe
        p_center_color = (
            f" {{\\color{{accentColor}}$|$}} "
            f"\\hrefWithoutArrow{{{portfolio_url}}}{{{pd}}}"
        )
        # template_06_minimal_mono: tabular header — plain pipe on same line
        p_center_plain = f"~~|~~\\hrefWithoutArrow{{{portfolio_url}}}{{{pd}}}"
        # AND-chain style (legacy, kept for any template still using it)
        p_mono = (
            f"\n        \\kern 5pt\\AND\\kern 5pt"
            f"\\mbox{{\\hrefWithoutArrow{{{portfolio_url}}}{{{pd}}}}}"
        )
        # template_04_bold_header: gold dot separator, white text
        p_exec = (
            f" \\enspace\\textcolor{{goldAccent}}{{$\\cdot$}}\\enspace"
            f"\\hrefWithoutArrow{{{portfolio_url}}}{{\\textcolor{{white}}{{{pd}}}}}"
        )
    else:
        p_white = p_white_sidebar = p_center = p_center_color = p_center_plain = p_mono = p_exec = ""

    tex = tex.replace("{{PORTFOLIO_HEADER_WHITE}}",  p_white)
    tex = tex.replace("{{PORTFOLIO_CENTER}}",         p_center)
    tex = tex.replace("{{PORTFOLIO_CENTER_COLOR}}",   p_center_color)
    tex = tex.replace("{{PORTFOLIO_HEADER_MONO}}",    p_mono)
    tex = tex.replace("{{PORTFOLIO_HEADER}}",         p_mono)   # legacy alias
    tex = tex.replace("{{PORTFOLIO_HEADER_EXEC}}",    p_exec)

    # ── GitHub header — legacy AND-chain style (template_02 old ver) ─
    # New template_02 injects GitHub directly in sidebar, so this is a no-op.
    if github_url:
        github_header = (
            f"\n        \\kern 5pt\\AND\\kern 5pt"
            f"\\mbox{{\\hrefWithoutArrow{{{github_url}}}{{GitHub}}}}"
        )
    else:
        github_header = "%"
    tex = tex.replace("{{GITHUB_HEADER}}", github_header)

    # ── Content blocks by template slug ─────────────────────────────
    # Use startswith() to handle both short slugs ("template_09") and
    # full slugs ("template_09_classic_jake") from the DB.
    if slug.startswith("template_02"):
        # SIDEBAR TEMPLATE:
        # Left dark panel → sidebar builders for skills, education, certs
        # Right white panel → minipage-safe builders for experience, projects
        # Summary is injected directly via {{SUMMARY}} placeholder above
        tex = tex.replace("{{SKILLS_BLOCK}}",         _skills_sidebar(ai_content.get("skills", {})))
        tex = tex.replace("{{EDUCATION_BLOCK}}",      _education_sidebar(ai_content.get("education", [])))
        tex = tex.replace("{{CERTIFICATIONS_BLOCK}}", _certifications_sidebar(ai_content.get("certifications", [])))
        tex = tex.replace("{{EXPERIENCE_BLOCK}}",     _experience_minipage(ai_content.get("experience", [])))
        tex = tex.replace("{{PROJECTS_BLOCK}}",       _projects_minipage(ai_content.get("projects", [])))

    elif slug.startswith("template_03") or slug.startswith("template_09"):
        # CLASSIC: resumeSubheading tabularx layout
        tex = tex.replace("{{SKILLS_BLOCK}}",         _skills_classic(ai_content.get("skills", {})))
        tex = tex.replace("{{EDUCATION_BLOCK}}",      _education_classic(ai_content.get("education", [])))
        tex = tex.replace("{{EXPERIENCE_BLOCK}}",     _experience_classic(ai_content.get("experience", [])))
        tex = tex.replace("{{PROJECTS_BLOCK}}",       _projects_classic(ai_content.get("projects", [])))
        tex = tex.replace("{{CERTIFICATIONS_BLOCK}}", _certifications_classic(ai_content.get("certifications", [])))

    elif slug.startswith("template_05"):
        # TWO-COLUMN: minipage LEFT/RIGHT split, different placeholder names
        tex = tex.replace("{{EXPERIENCE_BLOCK_LEFT}}",      _experience_twocol(ai_content.get("experience", [])))
        tex = tex.replace("{{PROJECTS_BLOCK_LEFT}}",        _projects_twocol(ai_content.get("projects", [])))
        tex = tex.replace("{{EDUCATION_BLOCK_RIGHT}}",      _education_twocol(ai_content.get("education", [])))
        tex = tex.replace("{{SKILLS_BLOCK_RIGHT}}",         _skills_twocol(ai_content.get("skills", {})))
        tex = tex.replace("{{CERTIFICATIONS_BLOCK_RIGHT}}", _certifications_twocol(ai_content.get("certifications", [])))

    elif slug.startswith("template_06"):
        # MINIMAL MONO: the template already owns the outer paracol columns.
        # Use blocks that do not create nested paracol/twocolentry environments.
        tex = tex.replace("{{SKILLS_BLOCK}}",         _skills_paracol(ai_content.get("skills", {})))
        tex = tex.replace("{{EDUCATION_BLOCK}}",      _education_sidebar(ai_content.get("education", [])))
        tex = tex.replace("{{CERTIFICATIONS_BLOCK}}", _certifications_paracol(ai_content.get("certifications", [])))
        tex = tex.replace("{{EXPERIENCE_BLOCK}}",     _experience_minipage(ai_content.get("experience", [])))
        tex = tex.replace("{{PROJECTS_BLOCK}}",       _projects_minipage(ai_content.get("projects", [])))

    elif slug.startswith("template_11"):
        # WARM SIDEBAR: minipage layout, section* headings, no paracol
        # Left: summary + skills + education + certs (minipage-safe)
        # Right: experience + projects (minipage-safe)
        tex = tex.replace("{{SKILLS_BLOCK}}",         _skills_warm_sidebar(ai_content.get("skills", {})))
        tex = tex.replace("{{EDUCATION_BLOCK}}",      _education_warm_sidebar(ai_content.get("education", [])))
        tex = tex.replace("{{CERTIFICATIONS_BLOCK}}", _certifications_warm_sidebar(ai_content.get("certifications", [])))
        tex = tex.replace("{{EXPERIENCE_BLOCK}}",     _experience_minipage(ai_content.get("experience", [])))
        tex = tex.replace("{{PROJECTS_BLOCK}}",       _projects_minipage(ai_content.get("projects", [])))

    else:
        # PARACOL LAYOUT — all remaining templates use onecolentry/twocolentry:
        # template_01_modern_blue, template_04_bold_header, template_06_minimal_mono
        # template_07_blue_accent, template_08_teal_garamond, template_10_charter_clean
        # template_12_navy_uppercase, template_13_crimson_double
        # template_14_purple_tri, template_15_slate_ruled
        tex = tex.replace("{{SKILLS_BLOCK}}",         _skills_paracol(ai_content.get("skills", {})))
        tex = tex.replace("{{EDUCATION_BLOCK}}",      _education_paracol(ai_content.get("education", [])))
        tex = tex.replace("{{EXPERIENCE_BLOCK}}",     _experience_paracol(ai_content.get("experience", [])))
        tex = tex.replace("{{PROJECTS_BLOCK}}",       _projects_paracol(ai_content.get("projects", [])))
        tex = tex.replace("{{CERTIFICATIONS_BLOCK}}", _certifications_paracol(ai_content.get("certifications", [])))

    # ── Remove empty sections ────────────────────────────────────────
    if not ai_content.get("certifications"):
        tex = re.sub(
            r"\\section\{Certifications\}.*?(?=\\section|\\end\{document\})",
            "", tex, flags=re.DOTALL,
        )
    if not ai_content.get("experience"):
        tex = re.sub(
            r"\\section\{[^}]*Experience[^}]*\}.*?(?=\\section|\\end\{document\})",
            "", tex, flags=re.DOTALL,
        )

    # ── Strip any remaining unreplaced placeholders ──────────────────
    tex = re.sub(r"\{\{[A-Z_]+\}\}", "", tex)
    return tex


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — R2 upload
# ─────────────────────────────────────────────────────────────────────────────

def _upload_to_r2(pdf_bytes: bytes, user_id: str, resume_id: str) -> str:
    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY")
    secret_key = os.environ.get("R2_SECRET_KEY")
    bucket     = os.environ.get("R2_BUCKET", "nyrvexa-resumes")
    if not all([account_id, access_key, secret_key]):
        logger.info("[R2] Skipping — R2 not configured")
        return ""
    try:
        import boto3
        s3 = boto3.client("s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto")
        key = f"resumes/{user_id}/{resume_id}.pdf"
        s3.put_object(Bucket=bucket, Key=key, Body=pdf_bytes, ContentType="application/pdf")
        return f"https://{account_id}.r2.cloudflarestorage.com/{bucket}/{key}"
    except Exception as e:
        logger.warning(f"[R2] Upload failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def build_resume(
    form_data:   dict,
    template_id: str,
    user_id:     str,
    resume_id:   str,
) -> dict:
    """
    Full pipeline:
      1. AI generates resume content JSON (Groq → DeepSeek → Gemini fallback)
      2. Retry if word_count < 400
      3. Inject into LaTeX template
      4. Compile with Tectonic
      5. Upload to R2 (skipped in local dev)
      6. Return dict
    """
    template = ResumeTemplate.query.filter_by(slug=template_id, is_active=True).first()
    if not template:
        try:
            template = ResumeTemplate.query.filter_by(id=template_id, is_active=True).first()
        except Exception:
            template = None
    if not template:
        raise ValueError(f"Template '{template_id}' not found or inactive.")

    logger.info(f"[Builder] Generating resume content for user {user_id}")
    ai_content = _call_ai_generate(form_data)

    word_count = ai_content.get("word_count", 999)
    if isinstance(word_count, int) and word_count < 400:
        logger.info(f"[Builder] word_count={word_count} < 400 — retrying with expansion instruction")
        ai_content = _call_ai_generate(
            form_data,
            extra_instruction=(
                f"The previous attempt had only {word_count} words — that is too short. "
                "You MUST write more detail in every bullet point. "
                "Each bullet should be a complete sentence with action + task + result. "
                "Add more specific technical details to project descriptions. "
                "Do NOT write short vague bullets. Aim for word_count between 500 and 560."
            ),
        )

    logger.info(f"[Builder] Injecting into template slug='{template.slug}'")
    latex_code = inject_placeholders(template, ai_content, form_data)

    logger.info(f"[Builder] Compiling LaTeX → PDF")
    pdf_bytes = compile_latex(latex_code)

    pdf_url = _upload_to_r2(pdf_bytes, user_id, resume_id)

    logger.info(f"[Builder] Done. PDF size={len(pdf_bytes)} bytes, url='{pdf_url}'")
    return {
        "pdf_bytes":  pdf_bytes,
        "pdf_url":    pdf_url,
        "ai_content": ai_content,
        "latex_code": latex_code,
    }


def enhance_bullets(bullets: list) -> list:
    """
    Rewrite raw bullet points into strong ATS-optimized sentences.
    Uses fallback chain: Groq Resume Key → Groq Main Key → DeepSeek.
    """
    system_prompt = (
        "You are an expert resume writer. Rewrite each bullet point into a strong, "
        "ATS-optimized sentence using: action verb + specific task + quantified result. "
        "Each rewritten bullet must be ONE sentence only. Max 20 words. "
        "Return ONLY a JSON array of rewritten strings, same count as input. No markdown."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Bullets to rewrite:\n{json.dumps(bullets)}"},
    ]

    providers = []
    grk = GROQ_RESUME_KEY()
    gmk = GROQ_MAIN_KEY()
    dsk = DEEPSEEK_KEY()
    if grk:
        providers.append((GROQ_URL, grk, "llama-3.3-70b-versatile"))
    if gmk and gmk != grk:
        providers.append((GROQ_URL, gmk, "llama-3.3-70b-versatile"))
    if dsk:
        providers.append((DEEPSEEK_URL, dsk, "deepseek-chat"))

    for url, key, model in providers:
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 1024,
            }
            resp = req.post(
                url, json=payload,
                headers={"Authorization": f"Bearer {key}"},
                timeout=45,
            )
            resp.raise_for_status()
            raw   = resp.json()["choices"][0]["message"]["content"]
            clean = re.sub(r"```json|```", "", raw).strip()
            return json.loads(clean)
        except Exception as e:
            logger.warning("[Enhance Bullets] Provider failed: %s", e)
            continue

    raise RuntimeError("[Enhance Bullets] All providers failed.")
