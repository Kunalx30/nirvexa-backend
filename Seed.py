import os, json
from app.extensions import db
from app.models.resume_template import ResumeTemplate

TEMPLATES_DIR = os.path.abspath('app/services/resume_templates')

for folder_name in sorted(os.listdir(TEMPLATES_DIR)):
    folder_path = os.path.join(TEMPLATES_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue
    meta_path = os.path.join(folder_path, 'metadata.json')
    tex_path = os.path.join(folder_path, 'template.tex')
    if not os.path.exists(meta_path) or not os.path.exists(tex_path):
        continue
    meta = json.load(open(meta_path, encoding='utf-8'))
    tex = open(tex_path, encoding='utf-8').read()
    slug = meta.get("id") or folder_name
    existing = ResumeTemplate.query.filter_by(slug=slug).first()
    if existing:
        existing.latex_code = tex
        existing.is_active = True
        print(f'[UPDATE] {slug}')
    else:
        db.session.add(ResumeTemplate(
            name=meta.get("name", slug),
            slug=slug,
            category=meta.get("category", "tech"),
            best_for=meta.get("best_for", ""),
            description=meta.get("description", ""),
            latex_code=tex,
            is_active=True
        ))
        print(f'[INSERT] {slug}')

db.session.commit()
print("Done.")