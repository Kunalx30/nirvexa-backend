import logging
from flask import Blueprint, jsonify, request
from app.models.news_cache import NewsCache

logger = logging.getLogger(__name__)
news_bp = Blueprint('news', __name__)

VALID_CATEGORIES = {'all', 'ai', 'data-science', 'startups', 'govt-jobs', 'general'}


@news_bp.route('/api/news', methods=['GET'])
def get_news():
    """
    GET /api/news
    Query params:
      - category: all | ai | data-science | startups | govt-jobs | general (default: all)
      - limit: int (default 20, max 50)
    """
    try:
        category = request.args.get('category', 'all').lower()
        limit    = min(int(request.args.get('limit', 20)), 50)

        if category not in VALID_CATEGORIES:
            return jsonify({'error': f'Invalid category. Choose from: {", ".join(VALID_CATEGORIES)}'}), 400

        query = NewsCache.query.order_by(NewsCache.published_at.desc())

        if category != 'all':
            query = query.filter_by(category=category)

        articles = query.limit(limit).all()

        return jsonify({
            'success':  True,
            'category': category,
            'count':    len(articles),
            'articles': [a.to_dict() for a in articles],
        }), 200

    except Exception as e:
        logger.error("[News Route] GET /api/news failed: %s", e)
        return jsonify({'error': 'Failed to fetch news'}), 500


@news_bp.route('/api/news/admin/trigger', methods=['GET'])
def trigger_news_pipeline():
    """
    Manual trigger for news pipeline.
    Protected by X-Admin-Secret header.
    """
    secret = request.headers.get('X-Admin-Secret')
    import os
    if secret != os.environ.get('ADMIN_SECRET', 'nirvexa-dev'):
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        from app.services.news_service import fetch_and_cache_news
        stats = fetch_and_cache_news()
        return jsonify({
            'success': True,
            'message': 'News pipeline completed',
            'stats':   stats,
        }), 200
    except Exception as e:
        logger.error("[News Admin] Trigger failed: %s", e)
        return jsonify({'error': str(e)}), 500