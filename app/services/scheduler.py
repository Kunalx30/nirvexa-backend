import logging
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler(timezone=pytz.utc)

# Stored at module level so run_daily_job_pipeline can access it
_app = None


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
    total_inserted = 0
    total_skipped = 0
    total_errors = 0

    # Step 1 — Scrape all sources in parallel (no DB, no app context needed)
    all_jobs = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_source = {
            executor.submit(fn): source
            for source, fn in scrapers.items()
        }
        for future in as_completed(future_to_source):
            source = future_to_source[future]
            try:
                jobs = future.result()
                all_jobs[source] = jobs
                logger.info(f"[{source}] scraped {len(jobs)} jobs")
            except Exception as e:
                logger.error(f"[{source}] Scrape error: {e}")
                all_jobs[source] = []

    # Step 2 — Insert all jobs inside app context
    with _app.app_context():
        for source, jobs in all_jobs.items():
            if not jobs:
                continue
            try:
                stats = insert_jobs(jobs)
                logger.info(
                    f"[{source}] inserted={stats['inserted']} "
                    f"skipped={stats['skipped']} "
                    f"errors={stats['errors']}"
                )
                total_inserted += stats['inserted']
                total_skipped  += stats['skipped']
                total_errors   += stats['errors']
            except Exception as e:
                logger.error(f"[{source}] Insert error: {e}")

    logger.info(
        f"=== Pipeline Done | "
        f"inserted={total_inserted} "
        f"skipped={total_skipped} "
        f"errors={total_errors} ==="
    )

    # Step 3 — Rebuild FAISS index with newly inserted jobs
    logger.info("[Scheduler] Rebuilding FAISS semantic search index...")
    try:
        from app.services.rag_pipeline import rebuild_index
        stats = rebuild_index(app=_app)
        logger.info(f"[Scheduler] FAISS rebuild complete: {stats}")
    except Exception as e:
        logger.error(f"[Scheduler] FAISS rebuild failed: {e}")


def init_scheduler(app):
    """Call this from create_app() to start the scheduler."""
    global _app
    _app = app  # store app reference for use in pipeline

    with app.app_context():
        scheduler.add_job(
            run_daily_job_pipeline,
            CronTrigger(hour=20, minute=30, timezone=pytz.utc),
            id='daily_job_pipeline',
            replace_existing=True,
            misfire_grace_time=3600,
        )
        scheduler.start()
        logger.info("APScheduler started — daily pipeline at 2AM IST")