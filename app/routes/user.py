"""
app/routes/user.py
NirVexa — User endpoints including saved jobs at /api/user/saved-jobs
"""
import logging
from flask import Blueprint, request, jsonify, g
from app.extensions import db
from app.models.job import Job
from app.models.saved_job import SavedJob
from app.middleware.auth_middleware import token_required
from app.routes.jobs import _serialize_job

from app.models.job_alert import JobAlert

logger = logging.getLogger(__name__)
user_bp = Blueprint("user", __name__, url_prefix="/api/user")


@user_bp.route("/saved-jobs", methods=["GET"])
@token_required
def get_saved_jobs():
    user_id = g.user_id
    saved = SavedJob.query.filter_by(user_id=user_id)\
        .order_by(SavedJob.saved_at.desc()).all()

    results = []
    for s in saved:
        job = Job.query.get(s.job_id)
        results.append({
            "id":       str(s.id),
            "job_id":   str(s.job_id),
            "status":   s.status,
            "notes":    s.notes,
            "saved_at": s.saved_at.isoformat() if s.saved_at else None,
            "job":      _serialize_job(job, include_description=False) if job else None,
        })

    return jsonify({"saved_jobs": results, "total": len(results)}), 200


@user_bp.route("/saved-jobs", methods=["POST"])
@token_required
def save_job():
    user_id = g.user_id
    data    = request.get_json() or {}
    job_id  = data.get("job_id")
    status  = data.get("status", "Saved")

    if not job_id:
        return jsonify({"error": "job_id required"}), 400

    job = Job.query.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    existing = SavedJob.query.filter_by(user_id=user_id, job_id=job_id).first()
    if existing:
        return jsonify({
            "id":     str(existing.id),
            "job_id": str(existing.job_id),
            "status": existing.status,
            "notes":  existing.notes,
        }), 200

    saved = SavedJob(user_id=user_id, job_id=job_id, status=status)
    db.session.add(saved)
    db.session.commit()

    return jsonify({
        "id":     str(saved.id),
        "job_id": str(saved.job_id),
        "status": saved.status,
        "notes":  saved.notes,
    }), 201


@user_bp.route("/saved-jobs/<saved_id>", methods=["PUT"])
@token_required
def update_saved_job(saved_id):
    user_id = g.user_id
    saved = SavedJob.query.filter_by(
        id=saved_id, user_id=user_id
    ).first_or_404()

    data = request.get_json() or {}
    if "status" in data:
        saved.status = data["status"]
    if "notes" in data:
        saved.notes = data["notes"]

    db.session.commit()

    return jsonify({
        "id":     str(saved.id),
        "job_id": str(saved.job_id),
        "status": saved.status,
        "notes":  saved.notes,
    }), 200


@user_bp.route("/saved-jobs/<saved_id>", methods=["DELETE"])
@token_required
def delete_saved_job(saved_id):
    user_id = g.user_id
    saved = SavedJob.query.filter_by(
        id=saved_id, user_id=user_id
    ).first_or_404()
    db.session.delete(saved)
    db.session.commit()
    return jsonify({"message": "Removed"}), 200

# ───────────────── JOB ALERTS ───────────────── #

@user_bp.route("/alerts", methods=["GET"])
@token_required
def get_alerts():
    user_id = g.user_id

    alerts = JobAlert.query.filter_by(user_id=user_id)\
        .order_by(JobAlert.created_at.desc()).all()

    return jsonify([
        {
            "id": str(a.id),
            "keywords": a.keywords,
            "location": a.location,
            "frequency": a.frequency,
            "is_active": a.is_active,
            "created_at": a.created_at.isoformat() if a.created_at else None
        }
        for a in alerts
    ]), 200


@user_bp.route("/alerts", methods=["POST"])
@token_required
def create_alert():
    user_id = g.user_id
    data = request.get_json() or {}

    alert = JobAlert(
        user_id=user_id,
        keywords=data.get("keywords", []),
        location=data.get("location", ""),
        frequency=data.get("frequency", "daily")
    )

    db.session.add(alert)
    db.session.commit()

    return jsonify({
        "id": str(alert.id),
        "keywords": alert.keywords,
        "location": alert.location,
        "frequency": alert.frequency,
        "is_active": alert.is_active
    }), 201


@user_bp.route("/alerts/<alert_id>", methods=["DELETE"])
@token_required
def delete_alert(alert_id):
    user_id = g.user_id

    alert = JobAlert.query.filter_by(
        id=alert_id,
        user_id=user_id
    ).first_or_404()

    db.session.delete(alert)
    db.session.commit()

    return jsonify({"message": "Deleted"}), 200