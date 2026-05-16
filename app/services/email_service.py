"""
app/services/email_service.py
NyrVexa — Emails via Brevo
"""
import os, logging, requests

logger = logging.getLogger(__name__)

BREVO_API_KEY = os.environ.get("BREVO_API_KEY")
FROM_EMAIL    = "team@nyrvexa.in"
FROM_NAME     = "Nyrvexa"


def _base(content: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>Nyrvexa</title>
</head>
<body style="margin:0;padding:0;background:#f3f3f3;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f3f3f3;padding:40px 16px;">
  <tr><td align="center">
  <table width="560" cellpadding="0" cellspacing="0" style="max-width:560px;width:100%;">

    <!-- Header -->
    <tr><td style="background:#fafafa;border:1px solid #e4e4e4;border-top:3px solid #0a0a0a;
                   border-radius:12px 12px 0 0;padding:28px 36px;text-align:center;">
      <div style="display:inline-block;">
        <div style="width:28px;height:28px;background:#0a0a0a;border-radius:7px;
                    display:inline-block;vertical-align:middle;font-size:13px;
                    font-weight:700;color:#fafafa;line-height:28px;text-align:center;">N</div>
        <span style="font-size:16px;font-weight:600;color:#0a0a0a;
                     letter-spacing:-0.4px;vertical-align:middle;margin-left:8px;">Nyrvexa</span>
      </div>
      <p style="margin:6px 0 0;font-size:12px;color:#6b6b6b;letter-spacing:0.5px;">AI-Powered Career Platform</p>
    </td></tr>

    <!-- Body -->
    <tr><td style="background:#fafafa;border:1px solid #e4e4e4;border-top:none;padding:36px;">
      {content}
    </td></tr>

    <!-- Footer -->
    <tr><td style="background:#f3f3f3;border:1px solid #e4e4e4;border-top:none;
                   border-radius:0 0 12px 12px;padding:20px 36px;text-align:center;">
      <p style="margin:0 0 6px;font-size:12px;color:#6b6b6b;">© 2026 Nyrvexa · Made with care in India</p>
      <p style="margin:0;font-size:12px;color:#6b6b6b;">
        <a href="https://www.nyrvexa.in" style="color:#0a0a0a;text-decoration:none;">nyrvexa.in</a>
        &nbsp;·&nbsp;
        <a href="mailto:team@nyrvexa.in" style="color:#0a0a0a;text-decoration:none;">team@nyrvexa.in</a>
      </p>
    </td></tr>

  </table>
  </td></tr>
</table>
</body>
</html>"""


def _divider() -> str:
    return '<div style="border-top:1px solid #e4e4e4;margin:28px 0;"></div>'


def _button(text: str, url: str) -> str:
    return f"""
    <div style="text-align:center;margin:0 0 32px;">
      <a href="{url}"
         style="display:inline-block;background:#0a0a0a;color:#fafafa;
                font-size:15px;font-weight:600;padding:13px 28px;
                border-radius:26px;text-decoration:none;letter-spacing:-0.3px;">
        {text}
      </a>
    </div>"""


def _send(to_email: str, to_name: str, subject: str, html: str) -> bool:
    try:
        resp = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={"api-key": BREVO_API_KEY, "Content-Type": "application/json"},
            json={
                "sender": {"name": FROM_NAME, "email": FROM_EMAIL},
                "to":     [{"email": to_email, "name": to_name}],
                "subject": subject,
                "htmlContent": html,
            },
            timeout=15,
        )
        resp.raise_for_status()
        logger.info(f"[Email] Sent '{subject}' to {to_email}")
        return True
    except Exception as e:
        logger.error(f"[Email] Failed to {to_email}: {e}")
        return False


def send_welcome_email(user_email: str, user_name: str) -> bool:
    content = f"""
      <p style="font-size:11px;font-weight:600;letter-spacing:1.4px;text-transform:uppercase;
                color:#6b6b6b;margin:0 0 12px;">Welcome aboard</p>

      <h1 style="font-size:32px;font-weight:400;letter-spacing:-1.5px;color:#0a0a0a;
                 margin:0 0 14px;line-height:1.05;">
        Your career platform<br>
        <em style="color:#6b6b6b;font-style:italic;">is ready.</em>
      </h1>

      <p style="font-size:15px;color:#6b6b6b;line-height:1.6;margin:0 0 28px;font-weight:400;">
        Hi {user_name}, you're now part of Nyrvexa — India's AI-powered career suite
        built for modern professionals. Everything you need to land your next role is right here.
      </p>

      {_button("Start exploring &rarr;", "https://www.nyrvexa.in")}
      {_divider()}

      <p style="font-size:13px;font-weight:600;color:#0a0a0a;margin:0 0 16px;letter-spacing:-0.2px;">
        Nine tools. One suite.
      </p>

        <table width="100%" cellpadding="0" cellspacing="0">
        <tr><td style="padding:10px 0;border-bottom:1px solid #e4e4e4;">
          <span style="font-size:13px;font-weight:600;color:#0a0a0a;">Hybrid AI Chat</span>
          <span style="font-size:12px;color:#6b6b6b;"> — Multiple frontier models routed to the best fit</span>
        </td></tr>
        <tr><td style="padding:10px 0;border-bottom:1px solid #e4e4e4;">
          <span style="font-size:13px;font-weight:600;color:#0a0a0a;">Real Job Listings</span>
          <span style="font-size:12px;color:#6b6b6b;"> — Live roles aggregated from top portals, direct apply links</span>
        </td></tr>
        <tr><td style="padding:10px 0;border-bottom:1px solid #e4e4e4;">
          <span style="font-size:13px;font-weight:600;color:#0a0a0a;">Resume Analyzer</span>
          <span style="font-size:12px;color:#6b6b6b;"> — ATS score, keyword gaps &amp; section-by-section suggestions</span>
        </td></tr>
        <tr><td style="padding:10px 0;border-bottom:1px solid #e4e4e4;">
          <span style="font-size:13px;font-weight:600;color:#0a0a0a;">Resume Tailor</span>
          <span style="font-size:12px;color:#6b6b6b;"> — Paste a JD, get your resume rewritten to beat the ATS</span>
        </td></tr>
        <tr><td style="padding:10px 0;border-bottom:1px solid #e4e4e4;">
          <span style="font-size:13px;font-weight:600;color:#0a0a0a;">Career Path AI</span>
          <span style="font-size:12px;color:#6b6b6b;"> — Skill gap analysis &amp; week-by-week learning roadmap</span>
        </td></tr>
        <tr><td style="padding:10px 0;border-bottom:1px solid #e4e4e4;">
          <span style="font-size:13px;font-weight:600;color:#0a0a0a;">Voice Interview AI</span>
          <span style="font-size:12px;color:#6b6b6b;"> — Speak your answers, scored on content, clarity &amp; confidence</span>
        </td></tr>
        <tr><td style="padding:10px 0;border-bottom:1px solid #e4e4e4;">
          <span style="font-size:13px;font-weight:600;color:#0a0a0a;">Company Research</span>
          <span style="font-size:12px;color:#6b6b6b;"> — Culture, funding, interview difficulty &amp; insider Q&amp;A</span>
        </td></tr>
        <tr><td style="padding:10px 0;border-bottom:1px solid #e4e4e4;">
          <span style="font-size:13px;font-weight:600;color:#0a0a0a;">Salary Insights</span>
          <span style="font-size:12px;color:#6b6b6b;"> — Real compensation by role, city, experience &amp; company</span>
        </td></tr>
        <tr><td style="padding:10px 0;">
          <span style="font-size:13px;font-weight:600;color:#0a0a0a;">Tech &amp; Career News</span>
          <span style="font-size:12px;color:#6b6b6b;"> — AI-summarised premium publications in under 60 seconds</span>
        </td></tr>
      </table>

      {_divider()}

      <p style="font-size:13px;color:#6b6b6b;margin:0;line-height:1.6;">
        Questions? Reply to this email or reach us at
        <a href="mailto:team@nyrvexa.in"
           style="color:#0a0a0a;font-weight:500;text-decoration:none;">team@nyrvexa.in</a>
      </p>
    """
    return _send(user_email, user_name,
                 "Welcome to Nyrvexa — your career platform is ready", _base(content))


def send_password_reset(user_email: str, user_name: str, reset_url: str) -> bool:
    content = f"""
      <p style="font-size:11px;font-weight:600;letter-spacing:1.4px;text-transform:uppercase;
                color:#6b6b6b;margin:0 0 12px;">Password reset</p>

      <h1 style="font-size:32px;font-weight:400;letter-spacing:-1.5px;color:#0a0a0a;
                 margin:0 0 14px;line-height:1.05;">
        Reset your<br>
        <em style="color:#6b6b6b;font-style:italic;">password.</em>
      </h1>

      <p style="font-size:15px;color:#6b6b6b;line-height:1.6;margin:0 0 28px;font-weight:400;">
        Hi {user_name}, we received a request to reset your Nyrvexa password.
        Click the button below to choose a new one. This link expires in 1 hour.
      </p>

      {_button("Reset my password &rarr;", reset_url)}
      {_divider()}

      <table width="100%" cellpadding="0" cellspacing="0"
             style="background:#f3f3f3;border:1px solid #e4e4e4;border-radius:8px;">
        <tr><td style="padding:16px 20px;">
          <p style="margin:0;font-size:13px;color:#6b6b6b;line-height:1.6;">
            If you didn't request a password reset, you can safely ignore this email.
            Your password will not change.
          </p>
        </td></tr>
      </table>
    """
    return _send(user_email, user_name,
                 "Nyrvexa — reset your password", _base(content))


def send_job_alert(user_email: str, user_name: str, jobs: list) -> bool:
    if not jobs:
        return False

    count  = len(jobs)
    plural = "s" if count > 1 else ""

    job_rows = ""
    for job in jobs[:10]:
        skills   = ", ".join(job.get("skills", [])[:4]) or "Not specified"
        salary   = job.get("salary") or "Not disclosed"
        company  = job.get("company", "N/A")
        title    = job.get("title", "N/A")
        location = job.get("location", "N/A")
        url      = job.get("apply_url", "https://www.nyrvexa.in/jobs")

        job_rows += f"""
        <table width="100%" cellpadding="0" cellspacing="0"
               style="border:1px solid #e4e4e4;border-left:3px solid #0a0a0a;
                      border-radius:8px;margin-bottom:16px;background:#fafafa;">
          <tr><td style="padding:20px;">
            <p style="margin:0 0 4px;font-size:15px;font-weight:600;
                      color:#0a0a0a;letter-spacing:-0.3px;">{title}</p>
            <p style="margin:0 0 12px;font-size:13px;color:#6b6b6b;">
              {company} &nbsp;·&nbsp; {location}
            </p>
            <p style="margin:0 0 2px;font-size:12px;color:#6b6b6b;">
              <span style="font-weight:600;color:#0a0a0a;">Skills:</span> {skills}
            </p>
            <p style="margin:0 0 16px;font-size:12px;color:#6b6b6b;">
              <span style="font-weight:600;color:#0a0a0a;">Salary:</span> {salary}
            </p>
            <a href="{url}"
               style="display:inline-block;background:#0a0a0a;color:#fafafa;
                      font-size:13px;font-weight:600;padding:8px 20px;
                      border-radius:20px;text-decoration:none;letter-spacing:-0.2px;">
              Apply now →
            </a>
          </td></tr>
        </table>
        """

    content = f"""
      <p style="font-size:11px;font-weight:600;letter-spacing:1.4px;text-transform:uppercase;
                color:#6b6b6b;margin:0 0 12px;">Job alert</p>

      <h1 style="font-size:32px;font-weight:400;letter-spacing:-1.5px;color:#0a0a0a;
                 margin:0 0 14px;line-height:1.05;">
        {count} new job{plural}<br>
        <em style="color:#6b6b6b;font-style:italic;">match your alert.</em>
      </h1>

      <p style="font-size:15px;color:#6b6b6b;line-height:1.6;margin:0 0 28px;font-weight:400;">
        Hi {user_name}, we found {count} new job{plural} matching your preferences.
        Good opportunities move fast — apply while they're live.
      </p>

      {_divider()}
      {job_rows}
      {_divider()}

      <p style="font-size:13px;color:#6b6b6b;margin:0;line-height:1.6;text-align:center;">
        Manage your alerts at
        <a href="https://www.nyrvexa.in/jobs"
           style="color:#0a0a0a;font-weight:500;text-decoration:none;">nyrvexa.in/jobs</a>
      </p>
    """
    return _send(user_email, user_name,
                 f"Nyrvexa — {count} new job{plural} matching your alert", _base(content))