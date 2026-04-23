import os
from flask import Blueprint, jsonify, request
from app.services.job_scraper import (
    scrape_remotive,
    scrape_github_jobs,
    scrape_internshala,
    scrape_greenhouse_companies,
    scrape_lever_companies,
    scrape_indian_startups_greenhouse,
    scrape_indian_startups_lever,
    scrape_jobicy_india,
    scrape_jsearch_india,
)
from app.services.job_inserter import insert_jobs
from app.models.job import Job

jobs_bp = Blueprint('jobs', __name__)


@jobs_bp.route('/admin/test-scrape', methods=['GET'])
def test_scrape():
    results = {}

    try:
        jobs = scrape_remotive()
        results['remotive'] = {'fetched': len(jobs), **insert_jobs(jobs)}
    except Exception as e:
        results['remotive'] = {'error': str(e)}

    try:
        jobs = scrape_github_jobs()
        results['arbeitnow'] = {'fetched': len(jobs), **insert_jobs(jobs)}
    except Exception as e:
        results['arbeitnow'] = {'error': str(e)}

    try:
        jobs = scrape_internshala()
        results['internshala'] = {'fetched': len(jobs), **insert_jobs(jobs)}
    except Exception as e:
        results['internshala'] = {'error': str(e)}

    try:
        jobs = scrape_greenhouse_companies()
        results['greenhouse_global'] = {'fetched': len(jobs), **insert_jobs(jobs)}
    except Exception as e:
        results['greenhouse_global'] = {'error': str(e)}

    try:
        jobs = scrape_lever_companies()
        results['lever_global'] = {'fetched': len(jobs), **insert_jobs(jobs)}
    except Exception as e:
        results['lever_global'] = {'error': str(e)}

    try:
        jobs = scrape_indian_startups_greenhouse()
        results['greenhouse_india'] = {'fetched': len(jobs), **insert_jobs(jobs)}
    except Exception as e:
        results['greenhouse_india'] = {'error': str(e)}

    try:
        jobs = scrape_indian_startups_lever()
        results['lever_india'] = {'fetched': len(jobs), **insert_jobs(jobs)}
    except Exception as e:
        results['lever_india'] = {'error': str(e)}

    try:
        jobs = scrape_jsearch_india()
        results['jsearch'] = {'fetched': len(jobs), **insert_jobs(jobs)}
    except Exception as e:
        results['jsearch'] = {'error': str(e)}

    try:
        jobs = scrape_jobicy_india()
        results['jobicy'] = {'fetched': len(jobs), **insert_jobs(jobs)}
    except Exception as e:
        results['jobicy'] = {'error': str(e)}

    return jsonify(results), 200


@jobs_bp.route('/admin/trigger-pipeline', methods=['POST'])
def trigger_pipeline():
    secret = request.headers.get('X-Admin-Secret')
    if secret != os.getenv('ADMIN_SECRET', 'nirvexa-admin-2026'):
        return jsonify({'error': 'Unauthorized'}), 401
    from app.services.scheduler import run_daily_job_pipeline
    run_daily_job_pipeline()
    return jsonify({'message': 'Pipeline triggered'}), 200


@jobs_bp.route('', methods=['GET'])
def get_jobs():
    jobs = (
        Job.query
        .filter_by(is_active=True)
        .order_by(Job.posted_at.desc())
        .limit(20)
        .all()
    )
    return jsonify([{
        'id': j.id,
        'title': j.title,
        'company': j.company,
        'location': j.location,
        'source': j.source,
        'job_type': j.job_type,
        'apply_url': j.apply_url,
        'posted_at': j.posted_at.isoformat() if j.posted_at else None,
    } for j in jobs]), 200