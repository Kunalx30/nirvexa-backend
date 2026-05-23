import logging
from flask import Blueprint, jsonify, request, g, current_app
from app.middleware.auth_middleware import token_required
from app.middleware.rate_limiter import rate_limit

logger = logging.getLogger(__name__)
career_bp = Blueprint('career', __name__)


@career_bp.route('/api/career/path', methods=['POST'])
@token_required
@rate_limit("career_roadmap")
def generate_career_path():
    """
    POST /api/career/path
    Body: {
        "current_role": "Data Analyst",
        "target_role": "ML Engineer",
        "current_skills": ["Python", "SQL", "Excel"],
        "experience_years": 2
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Request body is required'}), 400

        current_role = (data.get('current_role') or '').strip()
        target_role  = (data.get('target_role') or '').strip()

        if not current_role or not target_role:
            return jsonify({'error': 'current_role and target_role are required'}), 400

        current_skills   = data.get('current_skills', [])
        experience_years = int(data.get('experience_years', 0))

        # Sanitize skills list
        if not isinstance(current_skills, list):
            current_skills = []
        current_skills = [s.strip() for s in current_skills if isinstance(s, str) and s.strip()]

        from app.services.career_service import generate_career_path as gen_path
        result = gen_path(current_role, target_role, current_skills, experience_years)

        if not result['success']:
            return jsonify({'error': result['error']}), 500

        return jsonify({
            'success':      True,
            'current_role': current_role,
            'target_role':  target_role,
            'career_path':  result['data'],
        }), 200

    except Exception as e:
        logger.error("[Career Route] POST /api/career/path failed: %s", e)
        return jsonify({'error': 'Failed to generate career path'}), 500


@career_bp.route('/api/career/skill-gap', methods=['POST'])
@token_required
@rate_limit("skill_match")
def get_skill_gap():
    """
    POST /api/career/skill-gap
    Body: {
        "user_skills": ["Python", "SQL", "Excel"],
        "target_job_title": "Data Scientist"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Request body is required'}), 400

        target_job_title = (data.get('target_job_title') or '').strip()

        if not target_job_title:
            return jsonify({'error': 'target_job_title is required'}), 400

        user_skills = data.get('user_skills', [])
        if not isinstance(user_skills, list):
            user_skills = []
        user_skills = [s.strip() for s in user_skills if isinstance(s, str) and s.strip()]

        from app.services.career_service import generate_skill_gap
        result = generate_skill_gap(user_skills, target_job_title)

        if not result['success']:
            return jsonify({'error': result['error']}), 500

        return jsonify({
            'success':          True,
            'target_job_title': target_job_title,
            'skill_gap':        result['data'],
        }), 200

    except Exception as e:
        logger.error("[Career Route] POST /api/career/skill-gap failed: %s", e)
        return jsonify({'error': 'Failed to generate skill gap analysis'}), 500


@career_bp.route('/api/jobs/salary-insights', methods=['GET'])
@token_required
@rate_limit("salary_insights")
def get_salary_insights():
    """
    GET /api/jobs/salary-insights?role=Data+Analyst&location=Bangalore
    Hybrid: DB data + DeepSeek AI fallback for Indian market insights.
    No auth required — public endpoint.
    """
    try:
        role     = (request.args.get('role') or '').strip()
        location = (request.args.get('location') or '').strip()

        if not role:
            return jsonify({'error': 'role parameter is required'}), 400

        from app.models.job import Job
        import re, json, openai

        # ── Step 1: Query DB ──────────────────────────────────────────────────
        query = Job.query.filter(
            Job.title.ilike(f'%{role}%'),
            Job.is_active == True
        )
        if location:
            query = query.filter(Job.location.ilike(f'%{location}%'))
        jobs = query.limit(50).all()

        # ── Step 2: Try to extract clean LPA values from DB ───────────────────
        salary_values = []
        sources = set()

        for job in jobs:
            if not job.salary:
                continue

            raw = job.salary.lower()
            sources.add(job.source)

            # Only process if it looks like Indian LPA data
            # Must contain 'lpa', 'lac', 'lakh', or small numbers (1-100 range)
            is_indian = any(k in raw for k in ['lpa', 'lac', 'lakh', '₹', 'inr'])

            if not is_indian:
                continue

            cleaned = raw.replace(',', '').replace('₹', '').replace('inr', '')
            cleaned = cleaned.replace('lpa', '').replace('lac', '').replace('lakh', '').strip()
            numbers = re.findall(r'\d+(?:\.\d+)?', cleaned)

            if not numbers:
                continue

            nums = [float(n) for n in numbers]
            # Only accept realistic LPA values (1–200 LPA)
            nums = [n for n in nums if 1 <= n <= 200]

            if nums:
                salary_values.append(sum(nums) / len(nums))

        # ── Step 3: If enough clean data, return DB-based insights ────────────
        if len(salary_values) >= 5:
            salary_values.sort()
            count = len(salary_values)
            return jsonify({
                'success':        True,
                'source':         'db',
                'role':           role,
                'location':       location or 'All India',
                'min_salary_lpa': round(min(salary_values), 1),
                'max_salary_lpa': round(max(salary_values), 1),
                'avg_salary_lpa': round(sum(salary_values) / count, 1),
                'median_lpa':     round(
                    salary_values[count // 2] if count % 2 != 0
                    else (salary_values[count // 2 - 1] + salary_values[count // 2]) / 2, 1
                ),
                'sample_count':   count,
                'sources':        list(sources),
                'note':           'Based on real job listings in our database.'
            }), 200

        # ── Step 4: Fall back to DeepSeek AI for market knowledge ─────────────
        location_str = location or 'India'
        prompt = f"""You are a salary expert for the Indian job market with up-to-date knowledge.

Provide realistic salary insights for: {role} in {location_str}

Respond ONLY with valid JSON. No preamble, no markdown, no backticks.

{{
  "min_salary_lpa": <float, lowest realistic salary>,
  "max_salary_lpa": <float, highest realistic salary>,
  "avg_salary_lpa": <float, average salary>,
  "median_lpa": <float, median salary>,
  "fresher_lpa": <float, starting salary for 0-1 years>,
  "experienced_lpa": <float, salary for 5+ years>,
  "top_paying_companies": ["Company 1", "Company 2", "Company 3"],
  "salary_factors": ["factor 1", "factor 2"],
  "market_demand": "high | medium | low",
  "note": "one sentence about salary trend for this role in India"
}}

Rules:
- You MUST provide highly specific, varying, and realistic salary numbers for this exact role. Do NOT give generic safe numbers.
- All salary values must be realistic LPA (Lakhs Per Annum) figures for the Indian market.
- top_paying_companies must be real companies hiring for this role in India."""

        try:
            try:
                client = openai.OpenAI(
                    api_key=current_app.config["DEEPSEEK_API_KEY"],
                    base_url="https://api.deepseek.com",
                    timeout=60.0
                )
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a salary expert. Always respond with valid JSON only. No markdown, no backticks."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=512,
                    temperature=0.2,
                )
                raw = response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"[SalaryInsights] DeepSeek failed, trying Groq fallback: {e}")
                import groq
                groq_client = groq.Groq(api_key=current_app.config.get("GROQ_API_KEY", ""))
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a salary expert. Always respond with valid JSON only. No markdown, no backticks."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=512,
                    temperature=0.2,
                )
                raw = response.choices[0].message.content.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            ai_data = json.loads(raw)
            ai_data['success']      = True
            ai_data['source']       = 'ai'
            ai_data['role']         = role
            ai_data['location']     = location_str
            ai_data['sample_count'] = len(jobs)

            return jsonify(ai_data), 200

        except Exception as ai_error:
            logger.error("[SalaryInsights] AI fallback failed: %s", ai_error)
            return jsonify({
                'success': False,
                'error':   'Insufficient salary data and AI fallback failed. Try a different role.'
            }), 500

    except Exception as e:
        logger.error("[Career Route] GET /api/jobs/salary-insights failed: %s", e)
        return jsonify({'error': 'Failed to fetch salary insights'}), 500


@career_bp.route('/api/career/company-research', methods=['POST'])
@token_required
@rate_limit("company_research")
def company_research():
    """
    POST /api/career/company-research
    Body: {
        "company_name": "Google",
        "location": "Bangalore"   ← optional, helps AI pin the right office
    }
    No auth required — public endpoint.
    Cached per company name to avoid repeated API calls.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400

        company_name = (data.get('company_name') or '').strip()
        if not company_name:
            return jsonify({'error': 'company_name is required'}), 400

        location = (data.get('location') or '').strip()  # optional

        from app.services.career_service import get_company_research
        result = get_company_research(company_name, location=location)

        if not result['success']:
            return jsonify({'error': result['error']}), 500

        return jsonify({
            'success':      True,
            'company_name': company_name,
            'research':     result['data'],
        }), 200

    except Exception as e:
        logger.error("[Career Route] POST /api/career/company-research failed: %s", e)
        return jsonify({'error': 'Failed to fetch company research'}), 500
