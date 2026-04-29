"""
app/services/email_service.py
NirVexa — Emails via Brevo
"""
import os, logging, requests

logger = logging.getLogger(__name__)

BREVO_API_KEY = os.environ.get("BREVO_API_KEY")
FROM_EMAIL    = "nirvexa.ai@gmail.com"
FROM_NAME     = "NirVexa"


def _send(to_email: str, to_name: str, subject: str, html: str) -> bool:
    try:
        resp = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={
                "api-key": BREVO_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "sender":      {"name": FROM_NAME, "email": FROM_EMAIL},
                "to":          [{"email": to_email, "name": to_name}],
                "subject":     subject,
                "htmlContent": html,
            },
            timeout=15,
        )
        resp.raise_for_status()
        logger.info(f"[Email] Sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"[Email] Failed to {to_email}: {e}")
        return False


def send_password_reset(user_email: str, user_name: str, reset_url: str) -> bool:
    html = f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;padding:24px;color:#1a1a1a;">
        <div style="text-align:center;margin-bottom:24px;">
            <h1 style="color:#6c63ff;">NirVexa</h1>
            <p style="color:#777;font-size:14px;">Your AI Career Assistant</p>
        </div>
        <h2 style="font-size:18px;">Hi {user_name}, reset your password</h2>
        <p style="color:#555;font-size:14px;margin-bottom:24px;">
            Click the button below to reset your password. This link expires in 1 hour.
        </p>
        <div style="text-align:center;margin-bottom:32px;">
            <a href="{reset_url}"
               style="background:#6c63ff;color:#fff;padding:12px 32px;border-radius:8px;
                      text-decoration:none;font-size:15px;font-weight:bold;">
               Reset Password
            </a>
        </div>
        <p style="color:#aaa;font-size:13px;">If you didn't request this, ignore this email.</p>
    </body>
    </html>
    """
    return _send(user_email, user_name, "NirVexa — Reset your password", html)


def send_job_alert(user_email: str, user_name: str, jobs: list) -> bool:
    if not jobs:
        return False

    job_cards = ""
    for job in jobs[:10]:
        skills = ", ".join(job.get("skills", [])[:5]) or "Not specified"
        salary = job.get("salary") or "Not disclosed"
        job_cards += f"""
        <div style="background:#f9f9f9;border-radius:8px;padding:16px;margin-bottom:16px;border-left:4px solid #6c63ff;">
            <h3 style="margin:0 0 4px;color:#1a1a1a;">{job.get('title', 'N/A')}</h3>
            <p style="margin:0 0 4px;color:#555;">{job.get('company', 'N/A')} — {job.get('location', 'N/A')}</p>
            <p style="margin:0 0 4px;color:#777;font-size:13px;">Skills: {skills}</p>
            <p style="margin:0 0 12px;color:#777;font-size:13px;">Salary: {salary}</p>
            <a href="{job.get('apply_url', '#')}"
               style="background:#6c63ff;color:#fff;padding:8px 16px;border-radius:6px;
                      text-decoration:none;font-size:13px;">Apply Now</a>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;padding:24px;color:#1a1a1a;">
        <div style="text-align:center;margin-bottom:24px;">
            <h1 style="color:#6c63ff;">NirVexa</h1>
        </div>
        <h2>Hi {user_name}, here are your new job matches!</h2>
        <p style="color:#555;font-size:14px;margin-bottom:24px;">
            We found {len(jobs)} new job{'s' if len(jobs) > 1 else ''} matching your alert.
        </p>
        {job_cards}
        <p style="color:#aaa;font-size:12px;text-align:center;margin-top:32px;">
            Manage your alerts by logging into NirVexa.
        </p>
    </body>
    </html>
    """

    subject = f"NirVexa — {len(jobs)} new job{'s' if len(jobs) > 1 else ''} matching your alert"
    return _send(user_email, user_name, subject, html)