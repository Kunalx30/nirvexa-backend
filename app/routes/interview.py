"""
app/routes/interview.py
NirVexa — Phase 6.6 + 6B: Text Interview Prep + Voice Interview AI
"""
import logging
from flask import Blueprint, jsonify, request, g
from app.middleware.auth_middleware import token_required

logger = logging.getLogger(__name__)
interview_bp = Blueprint('interview', __name__)


# ── Phase 6.6 — Text Interview Prep ──────────────────────────────────────────

@interview_bp.route('/text-prep', methods=['POST'])
@token_required
def text_prep():
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
        logger.error("[Interview Route] POST /text-prep failed: %s", e)
        return jsonify({'error': 'Failed to generate interview questions'}), 500


@interview_bp.route('/text-evaluate', methods=['POST'])
@token_required
def text_evaluate():
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
        logger.error("[Interview Route] POST /text-evaluate failed: %s", e)
        return jsonify({'error': 'Failed to evaluate answer'}), 500


# ── Phase 6B — Voice Interview AI ────────────────────────────────────────────

@interview_bp.route('/generate', methods=['POST'])
@token_required
def generate():
    """
    POST /api/interview/generate
    Body: { "role": str, "mode": "hr|technical|stress|mock", "difficulty": "easy|medium|hard" }
    Returns: { questions: [str], session_config: { role, mode, question_count } }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400

        role       = (data.get('role') or '').strip()
        mode       = (data.get('mode') or 'hr').strip().lower()
        difficulty = (data.get('difficulty') or 'medium').strip().lower()

        if not role:
            return jsonify({'error': 'role is required'}), 400
        if mode not in ('hr', 'technical', 'stress', 'mock'):
            return jsonify({'error': 'mode must be hr, technical, stress, or mock'}), 400
        if difficulty not in ('easy', 'medium', 'hard'):
            return jsonify({'error': 'difficulty must be easy, medium, or hard'}), 400

        from app.services.interview_service import generate_voice_questions
        result = generate_voice_questions(role, mode, difficulty)

        if not result['success']:
            return jsonify({'error': result['error']}), 500

        return jsonify({
            'success':        True,
            'questions':      result['questions'],
            'session_config': {
                'role':           role,
                'mode':           mode,
                'difficulty':     difficulty,
                'question_count': len(result['questions']),
            },
        }), 200

    except Exception as e:
        logger.error("[Interview Route] POST /generate failed: %s", e)
        return jsonify({'error': 'Failed to generate interview questions'}), 500


@interview_bp.route('/evaluate', methods=['POST'])
@token_required
def evaluate():
    """
    POST /api/interview/evaluate
    Body: {
        "question": str,
        "transcript": str,
        "role": str,
        "duration_seconds": int,
        "word_count": int,
        "filler_count": int
    }
    Returns: full 6-metric evaluation JSON
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400

        question         = (data.get('question') or '').strip()
        transcript       = (data.get('transcript') or '').strip()
        role             = (data.get('role') or '').strip()
        duration_seconds = int(data.get('duration_seconds') or 0)
        word_count       = int(data.get('word_count') or 0)
        filler_count     = int(data.get('filler_count') or 0)

        if not question or not transcript:
            return jsonify({'error': 'question and transcript are required'}), 400
        if not role:
            return jsonify({'error': 'role is required'}), 400
        if len(transcript) < 10:
            return jsonify({'error': 'Transcript is too short to evaluate'}), 400

        from app.services.interview_service import evaluate_voice_answer
        result = evaluate_voice_answer(
            question, transcript, role,
            duration_seconds, word_count, filler_count
        )

        if not result['success']:
            return jsonify({'error': result['error']}), 500

        return jsonify({
            'success':    True,
            'question':   question,
            'evaluation': result['data'],
        }), 200

    except Exception as e:
        logger.error("[Interview Route] POST /evaluate failed: %s", e)
        return jsonify({'error': 'Failed to evaluate answer'}), 500


@interview_bp.route('/session', methods=['POST'])
@token_required
def create_session():
    """
    POST /api/interview/session
    Body: { "role": str, "mode": str, "difficulty": str, "question_count": int }
    Returns: { session_id: UUID }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400

        role           = (data.get('role') or '').strip()
        mode           = (data.get('mode') or 'hr').strip()
        difficulty     = (data.get('difficulty') or 'medium').strip()
        question_count = int(data.get('question_count') or 10)

        if not role:
            return jsonify({'error': 'role is required'}), 400

        from app.services.interview_service import create_interview_session
        result = create_interview_session(g.user_id, role, mode, difficulty, question_count)

        if not result['success']:
            return jsonify({'error': result['error']}), 500

        return jsonify({
            'success':    True,
            'session_id': result['session_id'],
        }), 201

    except Exception as e:
        logger.error("[Interview Route] POST /session failed: %s", e)
        return jsonify({'error': 'Failed to create session'}), 500


@interview_bp.route('/session/<session_id>', methods=['PUT'])
@token_required
def complete_session(session_id):
    """
    PUT /api/interview/session/:id
    Body: { "total_score": float, "avg_wpm": float, "filler_word_count": int }
    Marks session as completed with final scores.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400

        total_score      = float(data.get('total_score') or 0)
        avg_wpm          = float(data.get('avg_wpm') or 0)
        filler_word_count = int(data.get('filler_word_count') or 0)

        from app.services.interview_service import finalize_interview_session
        result = finalize_interview_session(session_id, g.user_id, total_score, avg_wpm, filler_word_count)

        if not result['success']:
            return jsonify({'error': result['error']}), 500

        return jsonify({'success': True, 'message': 'Session completed'}), 200

    except Exception as e:
        logger.error("[Interview Route] PUT /session/%s failed: %s", session_id, e)
        return jsonify({'error': 'Failed to complete session'}), 500


@interview_bp.route('/sessions', methods=['GET'])
@token_required
def get_sessions():
    """
    GET /api/interview/sessions
    Returns all past sessions for the logged-in user, newest first.
    """
    try:
        from app.services.interview_service import get_user_sessions
        result = get_user_sessions(g.user_id)

        if not result['success']:
            return jsonify({'error': result['error']}), 500

        return jsonify({
            'success':  True,
            'sessions': result['sessions'],
        }), 200

    except Exception as e:
        logger.error("[Interview Route] GET /sessions failed: %s", e)
        return jsonify({'error': 'Failed to fetch sessions'}), 500


@interview_bp.route('/session/<session_id>', methods=['GET'])
@token_required
def get_session(session_id):
    """
    GET /api/interview/session/:id
    Returns full session report with all responses.
    """
    try:
        from app.services.interview_service import get_session_report
        result = get_session_report(session_id, g.user_id)

        if not result['success']:
            return jsonify({'error': result['error']}), 404

        return jsonify({
            'success': True,
            'session': result['session'],
        }), 200

    except Exception as e:
        logger.error("[Interview Route] GET /session/%s failed: %s", session_id, e)
        return jsonify({'error': 'Failed to fetch session'}), 500


@interview_bp.route('/session/<session_id>', methods=['DELETE'])
@token_required
def delete_session(session_id):
    """
    DELETE /api/interview/session/:id
    Deletes a session and all its responses.
    """
    try:
        from app.services.interview_service import delete_interview_session
        result = delete_interview_session(session_id, g.user_id)

        if not result['success']:
            return jsonify({'error': result['error']}), 404

        return jsonify({'success': True, 'message': 'Session deleted'}), 200

    except Exception as e:
        logger.error("[Interview Route] DELETE /session/%s failed: %s", session_id, e)
        return jsonify({'error': 'Failed to delete session'}), 500