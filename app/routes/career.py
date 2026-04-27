import logging
from flask import Blueprint, jsonify, request, g
from app.middleware.auth_middleware import token_required

logger = logging.getLogger(__name__)
career_bp = Blueprint('career', __name__)


@career_bp.route('/api/career/path', methods=['POST'])
@token_required
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

