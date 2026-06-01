"""
app/routes/jobs.py
NyrVexa Jobs API with FAISS Semantic Search
"""

import logging
import os
from datetime import datetime, timedelta, timezone

from flask import Blueprint, request, jsonify, current_app, g
from sqlalchemy import or_

from app.extensions import db
from app.models.job import Job
from app.models.admin_job import AdminJob
from app.models.saved_job import SavedJob
from app.models.job_alert import JobAlert
from app.models.user import User
from app.middleware.rate_limiter import get_current_user, _is_premium_active
from app.services.rag_pipeline import (
    search_jobs,
    match_jobs_by_skills,
    rebuild_index,
    get_index_status,
)
from app.middleware.auth_middleware import token_required
from app.middleware.rate_limiter import premium_required

logger = logging.getLogger(__name__)
jobs_bp = Blueprint("jobs", __name__)


JOB_TYPE_ALIASES = {
    "full-time": ["fulltime", "full-time", "full time", "full_time"],
    "fulltime": ["fulltime", "full-time", "full time", "full_time"],
    "full time": ["fulltime", "full-time", "full time", "full_time"],
    "contract": ["contract", "contractor", "contractual"],
    "contractor": ["contract", "contractor", "contractual"],
    "internship": ["internship", "intern", "trainee"],
    "intern": ["internship", "intern", "trainee"],
    "remote": ["remote"],
    "freelance": ["freelance", "freelancer"],
    "part-time": ["parttime", "part-time", "part time", "part_time"],
    "parttime": ["parttime", "part-time", "part time", "part_time"],
}


def _job_type_terms(value: str) -> list[str]:
    key = (value or "").strip().lower()
    return JOB_TYPE_ALIASES.get(key, [key] if key else [])


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
        "is_fresher":  job.is_fresher or False,
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
            "Summarize this job in exactly 3 short sentences for a job seeker. "
            "Cover: what the role involves, key skills needed, and one reason it's a good opportunity.\n\n"
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


@jobs_bp.route("", methods=["GET"])
def get_jobs():
    q            = request.args.get("q", "").strip()
    location     = request.args.get("location", "").strip()
    company      = request.args.get("company", "").strip()
    skills       = request.args.get("skills", "").strip()
    job_type     = request.args.get("type", "").strip()
    source       = request.args.get("source", "").strip()
    posted_within = request.args.get("posted_within", "").strip()
    fresher_only = request.args.get("fresher", "").strip().lower()
    page         = max(1, int(request.args.get("page", 1)))
    limit        = min(50, max(1, int(request.args.get("limit", 20))))
    offset       = (page - 1) * limit

    faiss_ids = []

    if q:
        faiss_ids = search_jobs(query=q, k=100)
        if faiss_ids:
            base_query = Job.query.filter(
                Job.id.in_(faiss_ids),
                Job.is_active == True,
            )
        else:
            logger.info("[Jobs] FAISS returned empty - falling back to SQL ILIKE for q='%s'", q)
            base_query = Job.query.filter(
                Job.is_active == True,
                or_(
                    Job.title.ilike(f"%{q}%"),
                    Job.company.ilike(f"%{q}%"),
                    Job.description.ilike(f"%{q}%"),
                ),
            )
    else:
        base_query = Job.query.filter(Job.is_active == True)

    if location:
        base_query = base_query.filter(Job.location.ilike(f"%{location}%"))
    if company:
        base_query = base_query.filter(Job.company.ilike(f"%{company}%"))
    if skills:
        skill_terms = [skill.strip().lower() for skill in skills.split(",") if skill.strip()]
        for term in skill_terms:
            base_query = base_query.filter(Job.skills.any(term))
    if posted_within:
        try:
            days = int(posted_within)
            if days > 0:
                cutoff = datetime.now(timezone.utc) - timedelta(days=days)
                base_query = base_query.filter(Job.posted_at >= cutoff)
        except ValueError:
            pass
    if job_type:
        terms = []
        for part in job_type.split(","):
            terms.extend(_job_type_terms(part.strip()))
        terms = [term for term in terms if term]
        if terms:
            type_conditions = [Job.job_type.ilike(f"%{term}%") for term in terms]
            if any(term in {"contract", "contractor", "contractual"} for term in terms):
                type_conditions.extend([
                    Job.title.ilike("%contract%"),
                    Job.description.ilike("%contract%"),
                ])
            base_query = base_query.filter(or_(*type_conditions))

    if fresher_only in {"true", "1", "yes"}:
        base_query = base_query.filter(Job.is_fresher == True)

    if source:
        sources = [src.strip().lower() for src in source.split(",") if src.strip()]
        if sources:
            from sqlalchemy import func
            if len(sources) == 1:
                base_query = base_query.filter(func.lower(Job.source) == sources[0])
            else:
                base_query = base_query.filter(func.lower(Job.source).in_(sources))

    if q and faiss_ids:
        from sqlalchemy import case
        order_map = case(
            {job_id: pos for pos, job_id in enumerate(faiss_ids)},
            value=Job.id,
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


def _serialize_premium_job(job: AdminJob, unlocked: bool) -> dict:
    base = {
        "id": job.id,
        "title": job.title,
        "company": job.company,
        "location": job.location,
        "job_type": job.job_type,
        "experience": job.experience,
        "salary": job.salary,
        "is_featured": job.is_featured,
        "category": job.category,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "source": "NirVexa Premium",
        "locked": not unlocked,
    }
    if unlocked:
        base.update({
            "company_logo": job.company_logo,
            "salary": job.salary,
            "description": job.description,
            "requirements": job.requirements,
            "skills": job.skills or [],
            "apply_url": job.apply_url,
            "apply_email": job.apply_email,
            "posted_by": job.posted_by,
        })
    return base


@jobs_bp.route("/premium", methods=["GET"])
@token_required
def get_premium_jobs():
    """Curated admin-posted jobs. Full details for Pro; preview for free users."""
    user = get_current_user() or User.query.get(str(g.user_id))
    unlocked = _is_premium_active(user)

    try:
        jobs = (
            AdminJob.query.filter_by(is_active=True)
            .order_by(AdminJob.is_featured.desc(), AdminJob.created_at.desc())
            .all()
        )
    except Exception as exc:
        logger.exception("[Premium Jobs] Query failed — ensuring table exists: %s", exc)
        from app.extensions import db
        db.create_all()
        jobs = (
            AdminJob.query.filter_by(is_active=True)
            .order_by(AdminJob.is_featured.desc(), AdminJob.created_at.desc())
            .all()
        )

    logger.info("[Premium Jobs] Returning %d active listing(s) (unlocked=%s)", len(jobs), unlocked)

    return jsonify({
        "jobs": [_serialize_premium_job(j, unlocked) for j in jobs],
        "unlocked": unlocked,
        "total": len(jobs),
    }), 200


@jobs_bp.route("/filters/options", methods=["GET"])
def get_filter_options():
    from sqlalchemy import func
    
    # Get distinct sources with counts
    sources_data = (
        db.session.query(Job.source, func.count(Job.id))
        .filter(Job.is_active == True, Job.source != None)
        .group_by(Job.source)
        .all()
    )
    source_counts = {s[0].strip(): s[1] for s in sources_data if s[0] and s[0].strip()}
    source_list = sorted(list(source_counts.keys()))

    # Get distinct types with counts
    types_data = (
        db.session.query(Job.job_type, func.count(Job.id))
        .filter(Job.is_active == True, Job.job_type != None)
        .group_by(Job.job_type)
        .all()
    )
    type_counts = {t[0].strip(): t[1] for t in types_data if t[0] and t[0].strip()}
    type_list = sorted(list(type_counts.keys()))

    # Total active jobs count
    total_active_jobs = db.session.query(func.count(Job.id)).filter(Job.is_active == True).scalar() or 0

    return jsonify({
        "sources": source_list,
        "types":   type_list,
        "source_counts": source_counts,
        "type_counts": type_counts,
        "total_jobs": total_active_jobs,
    }), 200



@jobs_bp.route("/<job_id>", methods=["GET"])
def get_job(job_id):
    job = Job.query.get_or_404(job_id)
    _generate_ai_summary(job)
    return jsonify(_serialize_job(job, include_description=True)), 200


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


@jobs_bp.route("/alerts", methods=["POST"])
@token_required
@premium_required("job_alerts")
def create_alert():
    data      = request.get_json() or {}
    keywords  = data.get("keywords", [])
    location  = data.get("location", "").strip()
    frequency = data.get("frequency", "daily")

    if not keywords:
        return jsonify({"error": "keywords required"}), 400

    alert = JobAlert(
        user_id=g.user_id,
        keywords=keywords if isinstance(keywords, list) else [keywords],
        location=location,
        frequency=frequency,
    )
    db.session.add(alert)
    db.session.commit()
    return jsonify({"message": "Alert created", "alert_id": str(alert.id)}), 201


@jobs_bp.route("/alerts", methods=["GET"])
@token_required
@premium_required("job_alerts")
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


def _check_admin(req) -> bool:
    secret = req.headers.get("X-Admin-Secret", "")
    return secret == os.environ.get("ADMIN_SECRET", "nyrvexa-dev")


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
