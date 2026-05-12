"""
app/scripts/seed_templates.py
Seeds ALL resume templates into the DB.
Run from flask shell:
    exec(open('app/scripts/seed_templates.py', encoding='utf-8').read())

Or paste the loop directly into flask shell if __file__ is unavailable.
"""
import os
import json
from app.extensions import db
from app.models.resume_template import ResumeTemplate

TEMPLATES_DIR = os.path.abspath('app/services/resume_templates')

loaded = 0
skipped = 0
errors = []

for folder_name in sorted(os.listdir(TEMPLATES_DIR)):
    folder_path = os.path.join(TEMPLATES_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue

    meta_path = os.path.join(folder_path, 'metadata.json')
    if not os.path.exists(meta_path):
        meta_path = os.path.join(folder_path, 'Metadata.json')
    tex_path  = os.path.join(folder_path, 'template.tex')

    if not os.path.exists(meta_path) or not os.path.exists(tex_path):
        print(f"  [SKIP] {folder_name} — missing metadata.json/Metadata.json or template.tex")
        skipped += 1
        continue

    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        with open(tex_path, 'r', encoding='utf-8') as f:
            tex = f.read()

        slug = meta.get("id") or folder_name
        existing = ResumeTemplate.query.filter_by(slug=slug).first()

        if existing:
            existing.name        = meta.get("name", slug)
            existing.category    = meta.get("category", "tech")
            existing.best_for    = meta.get("best_for", "")
            existing.description = meta.get("description", "")
            existing.latex_code  = tex
            existing.is_active   = True
            print(f"  [UPDATE] {slug}")
        else:
            record = ResumeTemplate(
                name        = meta.get("name", slug),
                slug        = slug,
                category    = meta.get("category", "tech"),
                best_for    = meta.get("best_for", ""),
                description = meta.get("description", ""),
                latex_code  = tex,
                is_active   = True,
            )
            db.session.add(record)
            print(f"  [INSERT] {slug}")

        loaded += 1

    except Exception as e:
        print(f"  [ERROR] {folder_name}: {e}")
        errors.append(folder_name)

db.session.commit()
print(f"\nDone. Loaded: {loaded}, Skipped: {skipped}, Errors: {len(errors)}")
if errors:
    print(f"Failed: {errors}")
