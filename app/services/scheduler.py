import logging
import os
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler(timezone=pytz.utc)

_app = None


def run_daily_job_pipeline():
    """
    Runs all scrapers in parallel, inserts results to DB,
    rebuilds FAISS index, and sends user alerts.
    Triggered daily at 2AM IST (20:30 UTC).
    """
    from app.services.job_inserter import insert_jobs
    from app.services.job_scraper import (
        scrape_remotive,
        scrape_github_jobs,
        scrape_internshala,
        scrape_lever_companies,
        scrape_indian_startups_greenhouse,
        scrape_indian_startups_lever,
        scrape_jsearch_india,
        scrape_remoteok,
        scrape_weworkremotely,
        scrape_workingnomads,
        scrape_himalayas,
        scrape_adzuna_by_city,
        scrape_adzuna_by_role,
         scrape_freshersworld,
        scrape_foundit_freshers,
        scrape_greenhouse_fresher_companies,
        scrape_wellfound_freshers,
    )

    scrapers = {
        'remotive':         scrape_remotive,
        'arbeitnow':        scrape_github_jobs,
        'internshala':      scrape_internshala,
        'lever_global':     scrape_lever_companies,
        'greenhouse_india': scrape_indian_startups_greenhouse,
        'lever_india':      scrape_indian_startups_lever,
        'remoteok':         scrape_remoteok,
        'weworkremotely':   scrape_weworkremotely,
        'workingnomads':    scrape_workingnomads,
        'himalayas':        scrape_himalayas,
         'freshersworld':        scrape_freshersworld,
        'foundit':              scrape_foundit_freshers,
        'greenhouse_freshers':  scrape_greenhouse_fresher_companies,
        'wellfound':            scrape_wellfound_freshers,

    }

    if os.getenv('RAPIDAPI_KEY'):
        scrapers['jsearch'] = scrape_jsearch_india

    if os.getenv('ADZUNA_APP_ID') and os.getenv('ADZUNA_APP_KEY'):
        scrapers['adzuna_city'] = scrape_adzuna_by_city
        scrapers['adzuna_role'] = scrape_adzuna_by_role

    logger.info("=== Daily Job Pipeline Started ===")

    # ── Step 1: Scrape all sources in parallel ────────────────────────────────
    all_jobs_by_source = {}
    newly_fetched_jobs = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_source = {
            executor.submit(fn): source
            for source, fn in scrapers.items()
        }
        for future in as_completed(future_to_source):
            source = future_to_source[future]
            try:
                jobs = future.result()
                all_jobs_by_source[source] = jobs
                newly_fetched_jobs.extend(jobs)
                logger.info("[%s] scraped %d jobs", source, len(jobs))
            except Exception as e:
                logger.error("[%s] Scraper error: %s", source, e)
                all_jobs_by_source[source] = []

    # ── Step 2: Insert all jobs inside app context ────────────────────────────
    total_inserted = 0
    total_skipped  = 0
    total_errors   = 0

    ctx = _app.app_context() if _app else None
    if ctx:
        ctx.push()

    try:
        for source, jobs in all_jobs_by_source.items():
            if not jobs:
                continue
            try:
                stats = insert_jobs(jobs)
                logger.info(
                    "[%s] inserted=%d skipped=%d errors=%d",
                    source, stats['inserted'], stats['skipped'], stats['errors']
                )
                total_inserted += stats['inserted']
                total_skipped  += stats['skipped']
                total_errors   += stats['errors']
            except Exception as e:
                logger.error("[%s] Insert error: %s", source, e)
    finally:
        if ctx:
            ctx.pop()

    logger.info(
        "=== Pipeline DB Sync Done | inserted=%d skipped=%d errors=%d ===",
        total_inserted, total_skipped, total_errors
    )

    # ── Step 3: Rebuild FAISS index ───────────────────────────────────────────
    logger.info("[Scheduler] Triggering FAISS rebuild...")
    try:
        from app.services.rag_pipeline import rebuild_index
        rebuild_index(app=_app)
    except Exception as e:
        logger.error("[Scheduler] FAISS rebuild failed: %s", e)

    # ── Step 4: Send Job Alerts ───────────────────────────────────────────────
    if total_inserted > 0:
        logger.info("[Scheduler] New jobs found. Processing alerts...")
        _send_job_alerts(newly_fetched_jobs)
    else:
        logger.info("[Scheduler] No new jobs inserted. Skipping alerts.")


def _send_job_alerts(new_jobs: list):
    """Match new jobs against user alerts and send emails."""
    from app.models.job_alert import JobAlert
    from app.models.user import User
    from app.services.email_service import send_job_alert
    from datetime import datetime, timezone

    if not _app:
        return

    with _app.app_context():
        alerts = JobAlert.query.filter_by(is_active=True).all()
        logger.info(f"[Alerts] Checking {len(alerts)} active alerts against {len(new_jobs)} new jobs")

        for alert in alerts:
            keywords = alert.keywords_list()
            if not keywords:
                continue

            matched = []
            for job in new_jobs:
                title = (job.get("title") or "").lower()
                skills = " ".join(job.get("skills") or []).lower()
                location = (job.get("location") or "").lower()
                alert_location = (alert.location or "").lower()

                keyword_match = any(kw in title or kw in skills for kw in keywords)
                location_match = not alert_location or alert_location in location

                if keyword_match and location_match:
                    matched.append(job)

            if not matched:
                continue

            user = User.query.filter_by(id=alert.user_id).first()
            if not user or not user.email:
                continue

            sent = send_job_alert(user.email, user.name, matched)
            if sent:
                alert.last_sent_at = datetime.now(timezone.utc)

        try:
            from app.database.db import db
            db.session.commit()
            logger.info("[Alerts] Alerts processed and timestamps updated.")
        except Exception as e:
            logger.error(f"[Alerts] DB commit failed: {e}")


def run_news_pipeline():
    """Fetch and cache news articles every 4 hours."""
    from app.services.news_service import fetch_and_cache_news

    logger.info("=== News Pipeline Started ===")
    ctx = _app.app_context() if _app else None
    if ctx:
        ctx.push()
    try:
        stats = fetch_and_cache_news()
        logger.info("=== News Pipeline Done | %s ===", stats)
    except Exception as e:
        logger.error("[News Pipeline] Failed: %s", e)
    finally:
        if ctx:
            ctx.pop()


def run_cleanup():
    """Delete jobs older than 30 days every night."""
    from app.models.job import Job
    from app.models.saved_job import SavedJob
    from app.database.db import db
    from datetime import datetime, timedelta, timezone

    with _app.app_context():
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)

        # Get IDs of old jobs NOT saved by any user
        old_jobs = (
            Job.query
            .filter(Job.posted_at < cutoff)
            .outerjoin(SavedJob, Job.id == SavedJob.job_id)
            .filter(SavedJob.job_id == None)
            .all()
        )

        if not old_jobs:
            logger.info("[Cleanup] No old unsaved jobs to delete.")
            return

        old_ids = [j.id for j in old_jobs]
        Job.query.filter(Job.id.in_(old_ids)).delete(synchronize_session=False)
        db.session.commit()
        logger.info(f"[Cleanup] Deleted {len(old_ids)} jobs older than 30 days (saved jobs preserved).")


def init_scheduler(app):
    """Call this from create_app() to start the scheduler."""
    global _app
    _app = app

    # 2AM IST = 20:30 UTC
    scheduler.add_job(
        run_daily_job_pipeline,
        CronTrigger(hour=20, minute=30, timezone=pytz.utc),
        id='daily_job_pipeline',
        replace_existing=True,
        misfire_grace_time=3600,
    )

    # 2:30AM IST = 21:00 UTC
    scheduler.add_job(
        run_cleanup,
        CronTrigger(hour=21, minute=0, timezone=pytz.utc),id='nightly_cleanup',
        replace_existing=True,
        misfire_grace_time=3600,
    )

    # News every 4 hours
    scheduler.add_job(
        run_news_pipeline,
        IntervalTrigger(hours=4),
        id='news_refresh_pipeline',
        replace_existing=True,
        misfire_grace_time=600,
    )

    scheduler.start()
    logger.info("APScheduler started — daily pipeline at 2AM IST")
