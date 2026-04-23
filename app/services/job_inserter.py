import logging
from app.extensions import db
from app.models.job import Job

logger = logging.getLogger(__name__)


def insert_jobs(jobs: list[dict]) -> dict:
    inserted = 0
    skipped = 0
    errors = 0

    for job in jobs:
        try:
            # Check duplicate BEFORE touching session
            exists = Job.query.filter_by(
                title=job['title'],
                company=job['company'],
                location=job['location'],
            ).first()

            if exists:
                skipped += 1
                continue

            new_job = Job(
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
            )
            db.session.add(new_job)

            # Flush ONE job at a time — catch constraint errors immediately
            db.session.flush()
            inserted += 1

        except Exception as e:
            db.session.rollback()  # Reset session after each failure
            if 'UniqueViolation' in str(e) or 'uq_job' in str(e):
                skipped += 1
            else:
                logger.error(f"Insert error for '{job.get('title')}': {e}")
                errors += 1

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Commit failed: {e}")

    return {'inserted': inserted, 'skipped': skipped, 'errors': errors}