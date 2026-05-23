"""
app/routes/roadmap_graph.py
New Blueprint — completely separate from career.py.
Register in your app factory:
    from app.routes.roadmap_graph import roadmap_graph_bp
    app.register_blueprint(roadmap_graph_bp)
"""
import logging
from flask import Blueprint, jsonify, request
from app.middleware.auth_middleware import token_required
from app.middleware.rate_limiter import rate_limit

logger = logging.getLogger(__name__)
roadmap_graph_bp = Blueprint('roadmap_graph', __name__)


@roadmap_graph_bp.route('/api/roadmap-graph/list', methods=['GET'])
def list_roadmaps():
    """GET /api/roadmap-graph/list — all available roadmaps"""
    try:
        from app.services.roadmap_graph_service import list_roadmaps
        return jsonify({'success': True, 'roadmaps': list_roadmaps()}), 200
    except Exception as e:
        logger.error("[RoadmapGraph] list failed: %s", e)
        return jsonify({'error': str(e)}), 500


@roadmap_graph_bp.route('/api/roadmap-graph/search', methods=['GET'])
@token_required
@rate_limit("roadmap_search")
def search_roadmaps():
    """GET /api/roadmap-graph/search?q=python"""
    try:
        q = (request.args.get('q') or '').strip()
        if not q:
            return jsonify({'error': 'q parameter required'}), 400
        from app.services.roadmap_graph_service import search_roadmaps
        return jsonify({'success': True, 'results': search_roadmaps(q)}), 200
    except Exception as e:
        logger.error("[RoadmapGraph] search failed: %s", e)
        return jsonify({'error': str(e)}), 500


@roadmap_graph_bp.route('/api/roadmap-graph/<roadmap_id>', methods=['GET'])
def get_roadmap(roadmap_id):
    """GET /api/roadmap-graph/python — full graph data"""
    try:
        from app.services.roadmap_graph_service import get_roadmap
        data = get_roadmap(roadmap_id)
        if not data:
            return jsonify({'error': f'Roadmap "{roadmap_id}" not found'}), 404
        return jsonify({'success': True, 'roadmap': data}), 200
    except Exception as e:
        logger.error("[RoadmapGraph] get failed: %s", e)
        return jsonify({'error': str(e)}), 500


@roadmap_graph_bp.route('/api/roadmap-graph/<roadmap_id>/flat', methods=['GET'])
@token_required
@rate_limit("roadmap_search")
def get_flat(roadmap_id):
    """GET /api/roadmap-graph/python/flat — flattened for AI use"""
    try:
        from app.services.roadmap_graph_service import get_flat_topics
        topics = get_flat_topics(roadmap_id)
        if not topics:
            return jsonify({'error': f'Roadmap "{roadmap_id}" not found'}), 404
        return jsonify({'success': True, 'topics': topics}), 200
    except Exception as e:
        logger.error("[RoadmapGraph] flat failed: %s", e)
        return jsonify({'error': str(e)}), 500
