"""
app/routes/interview.py
NirVexa — Phase 6.6: Text Interview Prep
"""

import logging
from flask import Blueprint, jsonify, request, g
from app.middleware.auth_middleware import token_required

logger = logging.getLogger(__name__)
interview_bp = Blueprint('interview', __name__)


@interview_bp.route('/text-prep', methods=['POST'])
@token_required
def text_prep():
    """
    POST /api/interview/text-prep
    Body: {
        "role": "Data Analyst",
        "type": "hr" | "technical"
    }
    Generates 10 interview questions for the given role and type.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400

        role      = (data.get('role') or '').strip()
        prep_type = (data.get('type') or 'hr').strip().lower()

        if not role:
            return jsonify({'error': 'role is required'}), 400

        if prep_type not in ('hr', 'technical'):
            return jsonify({'error': 'type must be hr or technical'}), 400

        from app.services.interview_service import generate_questions
        result = generate_questions(role, prep_type)

        if not result['success']:
            return jsonify({'error': result['error']}), 500

        return jsonify({
            'success':   True,
            'role':      role,
            'type':      prep_type,
            'questions': result['questions'],
        }), 200

    except Exception as e:
        logger.error("[Interview Route] POST /api/interview/text-prep failed: %s", e)
        return jsonify({'error': 'Failed to generate interview questions'}), 500


@interview_bp.route('/text-evaluate', methods=['POST'])
@token_required
def text_evaluate():
    """
    POST /api/interview/text-evaluate
    Body: {
        "question": "Tell me about yourself",
        "answer": "I am a data analyst with 2 years of experience...",
        "role": "Data Analyst"
    }
    Evaluates the written answer and returns score + feedback.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400

        question = (data.get('question') or '').strip()
        answer   = (data.get('answer') or '').strip()
        role     = (data.get('role') or '').strip()

        if not question or not answer:
            return jsonify({'error': 'question and answer are required'}), 400

        if not role:
            return jsonify({'error': 'role is required'}), 400

        if len(answer) < 10:
            return jsonify({'error': 'Answer is too short to evaluate'}), 400

        from app.services.interview_service import evaluate_answer
        result = evaluate_answer(question, answer, role)

        if not result['success']:
            return jsonify({'error': result['error']}), 500

        return jsonify({
            'success':    True,
            'question':   question,
            'evaluation': result['data'],
        }), 200

    except Exception as e:
        logger.error("[Interview Route] POST /api/interview/text-evaluate failed: %s", e)
        return jsonify({'error': 'Failed to evaluate answer'}), 500