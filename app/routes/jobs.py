"""
app/routes/jobs.py
NirVexa — Phase 5.6 + 5.7: Full Jobs API with FAISS Semantic Search
Kunal Chandelkar | April 2026
"""

import logging
import os
import re

from flask import Blueprint, request, jsonify, current_app, g
from sqlalchemy import or_

from app.extensions import db
from app.models.job import Job
from app.models.saved_job import SavedJob
from app.models.job_alert import JobAlert
from app.services.rag_pipeline import (
    search_jobs,
    match_jobs_by_skills,
    rebuild_index,
    get_index_status,
)
from app.middleware.auth_middleware import token_required

logger = logging.getLogger(__name__)
jobs_bp = Blueprint("jobs", __name__, url_prefix="/api/jobs")


# ── Helpers ───────────────────────────────────────────────────────

def _serialize_job(job: Job, include_description: bool = False) -> dict:
    data = {
        "id":          str(job.id),
        "title":       job.title,
        "company":     job.company,
        "location":    job.location,
        "skills":      job.skills or [],
        "salary":      job.salary,
        "source":      job.source,
        "job_type":    job.job_type,
        "apply_url":   job.apply_url,
        "posted_at":   job.posted_at.isoformat() if job.posted_at else None,
        "expires_at":  job.expires_at.isoformat() if job.expires_at else None,
        "ai_summary":  job.ai_summary,
    }
    if include_description:
        data["description"] = job.description
    return data


def _generate_ai_summary(job: Job) -> str | None:
    if job.ai_summary:
        return job.ai_summary
    try:
        from app.services.llm_router import get_llm_response
        api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            return None
        prompt = (
            f"Summarize this job in exactly 3 short sentences for a job seeker. "
            f"Cover: what the role involves, key skills needed, and one reason it's a good opportunity.\n\n"
            f"Job: {job.title} at {job.company} in {job.location}\n"
            f"Skills: {', '.join(job.skills or [])}\n"
            f"Description: {(job.description or '')[:500]}"
        )
        summary = get_llm_response(prompt, max_tokens=150)
        if summary:
            job.ai_summary = summary.strip()
            db.session.commit()
        return job.ai_summary
    except Exception as e:
        logger.warning("[Jobs] AI summary generation failed for job %s: %s", job.id, e)
        return None


# ── GET /api/jobs ─────────────────────────────────────────────────

@jobs_bp.route("", methods=["GET"])
def get_jobs():
    q        = request.args.get("q", "").strip()
    location = request.args.get("location", "").strip()
    job_type = request.args.get("type", "").strip()
    source   = request.args.get("source", "").strip()
    page     = max(1, int(request.args.get("page", 1)))
    limit    = min(50, max(1, int(request.args.get("limit", 20))))
    offset   = (page - 1) * limit

    faiss_ids = []

    if q:
        faiss_ids = search_jobs(query=q, k=100)
        if faiss_ids:
            base_query = Job.query.filter(
                Job.id.in_(faiss_ids),
                Job.is_active == True,
            )
        else:
            logger.info("[Jobs] FAISS returned empty — falling back to SQL ILIKE for q='%s'", q)
            base_query = Job.query.filter(
                Job.is_active == True,
                or_(
                    Job.title.ilike(f"%{q}%"),
                    Job.company.ilike(f"%{q}%"),
                    Job.description.ilike(f"%{q}%"),
                )
            )
    else:
        base_query = Job.query.filter(Job.is_active == True)

    if location:
        base_query = base_query.filter(Job.location.ilike(f"%{location}%"))
    if job_type:
        base_query = base_query.filter(Job.job_type.ilike(f"%{job_type}%"))
    if source:
        base_query = base_query.filter(Job.source == source)

    if q and faiss_ids:
        from sqlalchemy import case
        order_map = case(
            {job_id: pos for pos, job_id in enumerate(faiss_ids)},
            value=Job.id
        )
        base_query = base_query.order_by(order_map)
    else:
        base_query = base_query.order_by(Job.posted_at.desc())

    total = base_query.count()
    jobs  = base_query.offset(offset).limit(limit).all()

    return jsonify({
        "jobs":        [_serialize_job(j) for j in jobs],
        "total":       total,
        "page":        page,
        "limit":       limit,
        "pages":       (total + limit - 1) // limit,
        "search_mode": "semantic" if (q and faiss_ids) else ("sql_fallback" if q else "browse"),
    }), 200


# ── GET /api/jobs/<job_id> ────────────────────────────────────────

@jobs_bp.route("/<job_id>", methods=["GET"])
def get_job(job_id):
    job = Job.query.get_or_404(job_id)
    _generate_ai_summary(job)
    return jsonify(_serialize_job(job, include_description=True)), 200


# ── POST /api/jobs/match ──────────────────────────────────────────

@jobs_bp.route("/match", methods=["POST"])
@token_required
def match_jobs():
    data   = request.get_json() or {}
    skills = data.get("skills", [])

    if not skills or not isinstance(skills, list):
        return jsonify({"error": "skills must be a non-empty list"}), 400

    skills  = [str(s).strip() for s in skills if str(s).strip()][:20]
    matched = match_jobs_by_skills(skills=skills, k=10)

    if not matched:
        return jsonify({"matches": [], "message": "Search index not ready yet."}), 200

    job_map = {
        str(j.id): j
        for j in Job.query.filter(
            Job.id.in_([m["job_id"] for m in matched]),
            Job.is_active == True,
        ).all()
    }

    results = []
    for m in matched:
        job = job_map.get(m["job_id"])
        if job:
            serialized = _serialize_job(job)
            serialized["match_percentage"] = m["match_percentage"]
            results.append(serialized)

    return jsonify({"matches": results, "skills_used": skills}), 200


# ── GET /api/jobs/salary-insights ────────────────────────────────

@jobs_bp.route("/salary-insights", methods=["GET"])
def salary_insights():
    role     = request.args.get("role", "").strip()
    location = request.args.get("location", "").strip()

    if not role:
        return jsonify({"error": "role param required"}), 400

    query = Job.query.filter(Job.is_active == True, Job.title.ilike(f"%{role}%"))
    if location:
        query = query.filter(Job.location.ilike(f"%{location}%"))

    jobs = query.limit(200).all()

    salaries = []
    for job in jobs:
        if not job.salary:
            continue
        nums = re.findall(r"[\d,]+", job.salary.replace("L", "00000").replace("K", "000"))
        nums = [int(n.replace(",", "")) for n in nums if n.replace(",", "").isdigit()]
        if nums:
            salaries.extend(nums)

    if not salaries:
        return jsonify({
            "role": role, "location": location or "All India",
            "sample_count": len(jobs),
            "message": "Salary data not available for this role yet.",
        }), 200

    salaries.sort()
    mid = len(salaries) // 2

    return jsonify({
        "role":          role,
        "location":      location or "All India",
        "min_salary":    min(salaries),
        "max_salary":    max(salaries),
        "median_salary": salaries[mid],
        "avg_salary":    round(sum(salaries) / len(salaries)),
        "sample_count":  len(jobs),
        "sources":       list({j.source for j in jobs if j.source}),
    }), 200


# ── Saved Jobs ────────────────────────────────────────────────────

@jobs_bp.route("/saved", methods=["GET"])
@token_required
def get_saved_jobs():
    saved = SavedJob.query.filter_by(user_id=g.user_id).order_by(
        SavedJob.saved_at.desc()
    ).all()
    results = []
    for s in saved:
        job = Job.query.get(s.job_id)
        results.append({
            "saved_job_id": str(s.id),
            "status":       s.status,
            "notes":        s.notes,
            "saved_at":     s.saved_at.isoformat() if s.saved_at else None,
            "job":          _serialize_job(job) if job else None,
        })
    return jsonify({"saved_jobs": results}), 200


@jobs_bp.route("/saved", methods=["POST"])
@token_required
def save_job():
    data   = request.get_json() or {}
    job_id = data.get("job_id")
    status = data.get("status", "Saved")

    if not job_id:
        return jsonify({"error": "job_id required"}), 400

    job = Job.query.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    existing = SavedJob.query.filter_by(user_id=g.user_id, job_id=job_id).first()
    if existing:
        return jsonify({"error": "Job already saved", "saved_job_id": str(existing.id)}), 409

    saved = SavedJob(user_id=g.user_id, job_id=job_id, status=status)
    db.session.add(saved)
    db.session.commit()
    return jsonify({"message": "Job saved", "saved_job_id": str(saved.id)}), 201


@jobs_bp.route("/saved/<saved_job_id>", methods=["PUT"])
@token_required
def update_saved_job(saved_job_id):
    saved = SavedJob.query.filter_by(
        id=saved_job_id, user_id=g.user_id
    ).first_or_404()
    data = request.get_json() or {}
    if "status" in data:
        saved.status = data["status"]
    if "notes" in data:
        saved.notes = data["notes"]
    db.session.commit()
    return jsonify({"message": "Updated", "status": saved.status}), 200


@jobs_bp.route("/saved/<saved_job_id>", methods=["DELETE"])
@token_required
def delete_saved_job(saved_job_id):
    saved = SavedJob.query.filter_by(
        id=saved_job_id, user_id=g.user_id
    ).first_or_404()
    db.session.delete(saved)
    db.session.commit()
    return jsonify({"message": "Removed from saved jobs"}), 200


# ── Job Alerts ────────────────────────────────────────────────────

@jobs_bp.route("/alerts", methods=["POST"])
@token_required
def create_alert():
    data      = request.get_json() or {}
    keywords  = data.get("keywords", [])
    location  = data.get("location", "").strip()
    frequency = data.get("frequency", "daily")

    if not keywords:
        return jsonify({"error": "keywords required"}), 400

    alert = JobAlert(
        user_id   = g.user_id,
        keywords  = keywords if isinstance(keywords, list) else [keywords],
        location  = location,
        frequency = frequency,
    )
    db.session.add(alert)
    db.session.commit()
    return jsonify({"message": "Alert created", "alert_id": str(alert.id)}), 201


@jobs_bp.route("/alerts", methods=["GET"])
@token_required
def get_alerts():
    alerts = JobAlert.query.filter_by(user_id=g.user_id).all()
    return jsonify({
        "alerts": [
            {
                "id":           str(a.id),
                "keywords":     a.keywords,
                "location":     a.location,
                "frequency":    a.frequency,
                "last_sent_at": a.last_sent_at.isoformat() if a.last_sent_at else None,
            }
            for a in alerts
        ]
    }), 200


@jobs_bp.route("/alerts/<alert_id>", methods=["DELETE"])
@token_required
def delete_alert(alert_id):
    alert = JobAlert.query.filter_by(id=alert_id, user_id=g.user_id).first_or_404()
    db.session.delete(alert)
    db.session.commit()
    return jsonify({"message": "Alert deleted"}), 200


# ── Admin ─────────────────────────────────────────────────────────

def _check_admin(req) -> bool:
    secret = req.headers.get("X-Admin-Secret", "")
    return secret == os.environ.get("ADMIN_SECRET", "nirvexa-dev")


@jobs_bp.route("/admin/faiss-status", methods=["GET"])
def faiss_status():
    if not _check_admin(request):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(get_index_status()), 200


@jobs_bp.route("/admin/rebuild-index", methods=["POST"])
def admin_rebuild_index():
    if not _check_admin(request):
        return jsonify({"error": "Unauthorized"}), 401
    from app.services.rag_pipeline import load_index_from_disk
    result = load_index_from_disk()
    return jsonify(result), 200


@jobs_bp.route("/admin/build-faiss", methods=["POST"])
def build_faiss():
    if not _check_admin(request):
        return jsonify({"error": "Unauthorized"}), 401
    rebuild_index(app=current_app._get_current_object())
    return jsonify({"status": "FAISS rebuild started"}), 202


@jobs_bp.route("/admin/trigger-pipeline", methods=["POST"])
def trigger_pipeline():
    if not _check_admin(request):
        return jsonify({"error": "Unauthorized"}), 401
    try:
        from app.services.scheduler import run_daily_job_pipeline
        import threading
        t = threading.Thread(target=run_daily_job_pipeline, daemon=True)
        t.start()
        return jsonify({"message": "Pipeline triggered in background"}), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500