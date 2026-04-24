"""
app/services/email_service.py
NirVexa — Phase 5.9: Job Alert Emails via Resend
"""
import os, logging, requests

logger = logging.getLogger(__name__)

RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
FROM_EMAIL = "onboarding@resend.dev"
FROM_NAME = "NirVexa Jobs"


def send_job_alert(user_email: str, user_name: str, jobs: list) -> bool:
    """Send job alert email to a user with matching jobs."""
    if not jobs:
        return False

    subject = f"NirVexa — {len(jobs)} new job{'s' if len(jobs) > 1 else ''} matching your alert"
    html = _build_email_html(user_name, jobs)

    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from": f"{FROM_NAME} <{FROM_EMAIL}>",
                "to": [user_email],
                "subject": subject,
                "html": html,
            },
            timeout=15,
        )
        resp.raise_for_status()
        logger.info(f"[Email] Alert sent to {user_email} — {len(jobs)} jobs")
        return True
    except Exception as e:
        logger.error(f"[Email] Failed to send to {user_email}: {e}")
        return False


def _build_email_html(user_name: str, jobs: list) -> str:
    """Build HTML email body with job cards."""

    job_cards = ""
    for job in jobs[:10]:
        skills = ", ".join(job.get("skills", [])[:5]) or "Not specified"
        salary = job.get("salary") or "Not disclosed"
        job_cards += f"""
        <div style="background:#f9f9f9;border-radius:8px;padding:16px;margin-bottom:16px;border-left:4px solid #6c63ff;">
            <h3 style="margin:0 0 4px;color:#1a1a1a;font-size:16px;">{job.get('title', 'N/A')}</h3>
            <p style="margin:0 0 4px;color:#555;font-size:14px;">{job.get('company', 'N/A')} — {job.get('location', 'N/A')}</p>
            <p style="margin:0 0 4px;color:#777;font-size:13px;">Skills: {skills}</p>
            <p style="margin:0 0 12px;color:#777;font-size:13px;">Salary: {salary}</p>
            <a href="{job.get('apply_url', '#')}"
               style="background:#6c63ff;color:#fff;padding:8px 16px;border-radius:6px;text-decoration:none;font-size:13px;">
               Apply Now
            </a>
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;padding:24px;color:#1a1a1a;">
        <div style="text-align:center;margin-bottom:24px;">
            <h1 style="color:#6c63ff;font-size:24px;margin:0;">NirVexa</h1>
            <p style="color:#777;font-size:14px;margin:4px 0;">Your AI Job Search Assistant</p>
        </div>

        <h2 style="font-size:18px;">Hi {user_name}, here are your new job matches!</h2>
        <p style="color:#555;font-size:14px;margin-bottom:24px;">
            We found {len(jobs)} new job{'s' if len(jobs) > 1 else ''} matching your alert keywords.
        </p>

        {job_cards}

        <div style="text-align:center;margin-top:32px;padding-top:16px;border-top:1px solid #eee;">
            <p style="color:#aaa;font-size:12px;">
                You're receiving this because you set up a job alert on NirVexa.<br>
                To manage your alerts, log in to your NirVexa account.
            </p>
        </div>
    </body>
    </html>
    """