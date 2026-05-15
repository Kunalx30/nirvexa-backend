import logging
from app.extensions import db
from app.models.job import Job

logger = logging.getLogger(__name__)


def _ensure_connection():
    """Ping DB and dispose engine if connection is dead."""
    try:
        db.session.execute(db.text('SELECT 1'))
    except Exception:
        db.session.rollback()
        db.engine.dispose()


def _build_job_obj(job: dict) -> Job:
    """Single place to construct a Job ORM object from a dict."""
    return Job(
        title=job['title'],
        company=job['company'],
        location=job['location'],
        skills=job['skills'],
        salary=job['salary'],
        source=job['source'],
        job_type=job['job_type'],
        apply_url=job['apply_url'],
        description=job['description'],
        posted_at=job['posted_at'],
        expires_at=job['expires_at'],
        is_fresher=job.get('is_fresher', False),
    )


def insert_jobs(jobs: list[dict]) -> dict:
    inserted = 0
    skipped  = 0
    errors   = 0

    # Ensure connection is alive before starting the batch
    _ensure_connection()

    for job in jobs:
        try:
            exists = Job.query.filter_by(
                title=job['title'],
                company=job['company'],
                location=job['location'],
            ).first()

            if exists:
                skipped += 1
                continue

            db.session.add(_build_job_obj(job))
            db.session.flush()
            inserted += 1

        except Exception as e:
            db.session.rollback()

            err_str = str(e)
            if 'UniqueViolation' in err_str or 'uq_job' in err_str:
                skipped += 1

            elif 'server closed the connection' in err_str or 'OperationalError' in err_str:
                # DB dropped — reconnect and retry this one job
                logger.warning(f"[Inserter] DB connection lost — reconnecting...")
                try:
                    db.engine.dispose()
                    _ensure_connection()

                    # Retry the same job once
                    exists = Job.query.filter_by(
                        title=job['title'],
                        company=job['company'],
                        location=job['location'],
                    ).first()

                    if exists:
                        skipped += 1
                    else:
                        db.session.add(_build_job_obj(job))
                        db.session.flush()
                        inserted += 1
                        logger.info(f"[Inserter] Retry succeeded for '{job.get('title')}'")

                except Exception as retry_err:
                    db.session.rollback()
                    logger.error(f"[Inserter] Retry failed for '{job.get('title')}': {retry_err}")
                    errors += 1

            else:
                logger.error(f"Insert error for '{job.get('title')}': {e}")
                errors += 1

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Commit failed: {e}")

    return {'inserted': inserted, 'skipped': skipped, 'errors': errors}