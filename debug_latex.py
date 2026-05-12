"""
Run this from nirvexa-backend folder:
python debug_latex.py

It will show you the exact LaTeX being compiled for templates 04, 05, 06
so we can find where ACCENTCOLOR and Missing \begin{document} come from.
"""
import os, sys
sys.path.insert(0, os.getcwd())

from app import create_app
from app.models.resume_template import ResumeTemplate
from app.database.db import db
from app.services.resume_builder_service import inject_placeholders

app = create_app()

# Minimal fake data — enough to trigger all placeholders
FAKE_FORM = {
    "full_name": "Test User",
    "email": "test@example.com",
    "phone": "+91 9999999999",
    "location": "Nagpur, Maharashtra",
    "linkedin_url": "https://linkedin.com/in/testuser",
    "github_url": "https://github.com/testuser",
    "portfolio_url": "",
    "target_role": "Data Scientist",
    "experience_section_title": "Experience",
}

FAKE_AI = {
    "summary": "Test summary sentence one. Test summary sentence two. Test summary sentence three.",
    "experience": [
        {
            "company": "Test Company",
            "role": "Data Scientist",
            "duration": "Jan 2025 – Present",
            "location": "Nagpur",
            "bullets": [
                "Built ML models improving accuracy by 25 percent.",
                "Designed ETL pipelines processing 500K records daily.",
                "Created dashboards reducing reporting time by 40 percent.",
            ]
        }
    ],
    "projects": [
        {
            "name": "Test Project",
            "tech": "Python, Flask",
            "description": ["Built a test project serving 1000 users.", "Deployed on Vercel with 99 percent uptime."],
            "github_url": "https://github.com/testuser/project",
            "live_url": "",
        }
    ],
    "skills": {
        "Languages": ["Python", "SQL"],
        "Frameworks": ["Flask", "FastAPI"],
        "Tools": ["Git", "VS Code"],
        "Databases": ["PostgreSQL", "SQLite"],
        "Cloud": ["Vercel", "Render"],
    },
    "education": [
        {
            "degree": "B.Tech Computer Science",
            "institution": "Test College",
            "year": "2021 – 2025",
            "cgpa": "7.5 / 10",
            "coursework": "Machine Learning, Databases, Algorithms",
        }
    ],
    "certifications": [
        {"name": "Data Science Cert", "issuer": "IIT Guwahati", "date": "2026"}
    ],
    "word_count": 500,
}

with app.app_context():
    for slug in ["template_04_bold_header", "template_05_two_column", "template_06_minimal_mono"]:
        t = ResumeTemplate.query.filter_by(slug=slug).first()
        if not t:
            print(f"[NOT FOUND] {slug}")
            continue

        try:
            latex = inject_placeholders(t, FAKE_AI, FAKE_FORM)
            # Save to file so you can inspect it
            out_path = f"debug_{slug}.tex"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(latex)
            print(f"[OK] {slug} → saved to {out_path}")

            # Check for known problem strings
            if "ACCENTCOLOR" in latex.upper() and "accentColor" not in latex:
                print(f"  ⚠️  ACCENTCOLOR issue detected")
            if "\\begin{document}" not in latex:
                print(f"  ⚠️  Missing \\begin{{document}}")
            if "{{" in latex:
                import re
                remaining = re.findall(r"\{\{[A-Z_]+\}\}", latex)
                if remaining:
                    print(f"  ⚠️  Unreplaced placeholders: {set(remaining)}")
        except Exception as e:
            print(f"[ERROR] {slug}: {e}")