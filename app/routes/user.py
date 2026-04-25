"""
app/routes/user.py
NirVexa — User endpoints including saved jobs at /api/user/saved-jobs
"""
import logging
from flask import Blueprint, request, jsonify
from app.extensions import db
from app.models.job import Job
from app.models.saved_job import SavedJob
from app.middleware.auth_middleware import token_required
from app.routes.jobs import _serialize_job

logger = logging.getLogger(__name__)
user_bp = Blueprint("user", __name__, url_prefix="/api/user")


@user_bp.route("/saved-jobs", methods=["GET"])
@token_required
def get_saved_jobs(current_user):
    saved = SavedJob.query.filter_by(user_id=current_user.id)\
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
def save_job(current_user):
    data   = request.get_json() or {}
    job_id = data.get("job_id")
    status = data.get("status", "Saved")

    if not job_id:
        return jsonify({"error": "job_id required"}), 400

    job = Job.query.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    existing = SavedJob.query.filter_by(
        user_id=current_user.id, job_id=job_id
    ).first()
    if existing:
        return jsonify({
            "id":     str(existing.id),
            "job_id": str(existing.job_id),
            "status": existing.status,
            "notes":  existing.notes,
        }), 200

    saved = SavedJob(user_id=current_user.id, job_id=job_id, status=status)
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
def update_saved_job(current_user, saved_id):
    saved = SavedJob.query.filter_by(
        id=saved_id, user_id=current_user.id
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
def delete_saved_job(current_user, saved_id):
    saved = SavedJob.query.filter_by(
        id=saved_id, user_id=current_user.id
    ).first_or_404()
    db.session.delete(saved)
    db.session.commit()
    return jsonify({"message": "Removed"}), 200