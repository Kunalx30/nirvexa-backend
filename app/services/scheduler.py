import logging
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler(timezone=pytz.utc)

_app = None   # stored at init time so the pipeline thread can use it


def run_daily_job_pipeline():
    """
    Runs all scrapers in parallel, inserts results to DB.
    Triggered daily at 2AM IST (20:30 UTC).
    """
    from app.services.job_scraper import (
        scrape_remotive,
        scrape_github_jobs,
        scrape_internshala,
        scrape_greenhouse_companies,
        scrape_lever_companies,
        scrape_indian_startups_greenhouse,
        scrape_indian_startups_lever,
        scrape_jsearch_india,
    )
    from app.services.job_inserter import insert_jobs

    scrapers = {
        'remotive':          scrape_remotive,
        'arbeitnow':         scrape_github_jobs,
        'internshala':       scrape_internshala,
        'greenhouse_global': scrape_greenhouse_companies,
        'lever_global':      scrape_lever_companies,
        'greenhouse_india':  scrape_indian_startups_greenhouse,
        'lever_india':       scrape_indian_startups_lever,
        'jsearch':           scrape_jsearch_india,
    }

    logger.info("=== Daily Job Pipeline Started ===")

    # ── Step 1: Scrape all sources in parallel (no DB, no context needed) ──────
    all_jobs_by_source = {}
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
        "=== Pipeline Done | inserted=%d skipped=%d errors=%d ===",
        total_inserted, total_skipped, total_errors
    )

    # ── Step 3: Rebuild FAISS index in background (non-blocking) ─────────────
    logger.info("[Scheduler] Triggering FAISS rebuild in background...")
    try:
        from app.services.rag_pipeline import rebuild_index
        rebuild_index(app=_app)   # returns immediately — runs in its own thread
    except Exception as e:
        logger.error("[Scheduler] FAISS rebuild trigger failed: %s", e)


def init_scheduler(app):
    """Call this from create_app() to start the scheduler."""
    global _app
    _app = app

    with app.app_context():
        # 2AM IST = 20:30 UTC
        scheduler.add_job(
            run_daily_job_pipeline,
            CronTrigger(hour=20, minute=30, timezone=pytz.utc),
            id='daily_job_pipeline',
            replace_existing=True,
            misfire_grace_time=3600,
        )
        scheduler.start()
        logger.info("APScheduler started — daily pipeline at 2AM IST")