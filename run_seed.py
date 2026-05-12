import json
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

from app.extensions import db
from app.models.resume_template import ResumeTemplate
from config import config_map


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _database_url() -> str:
    return (
        os.getenv("SQLALCHEMY_DATABASE_URI")
        or os.getenv("DATABASE_URL")
        or ""
    ).strip()


def create_seed_app() -> Flask:
    env = os.getenv("FLASK_ENV", "development")
    app = Flask(__name__)
    app.config.from_object(config_map[env])

    db_url = _database_url()
    if not db_url:
        raise RuntimeError(
            "Database URL is missing. Add DATABASE_URL=... or "
            "SQLALCHEMY_DATABASE_URI=... to .env, then run this script again."
        )

    app.config["SQLALCHEMY_DATABASE_URI"] = db_url
    app.config.setdefault("SQLALCHEMY_TRACK_MODIFICATIONS", False)
    db.init_app(app)
    return app


def main() -> None:
    app = create_seed_app()
    templates_dir = BASE_DIR / "app" / "services" / "resume_templates"
    loaded = 0
    skipped = 0
    errors = []

    with app.app_context():
        for folder_path in sorted(p for p in templates_dir.iterdir() if p.is_dir()):
            meta_path = None
            for filename in ("metadata.json", "Metadata.json"):
                candidate = folder_path / filename
                if candidate.exists():
                    meta_path = candidate
                    break

            tex_path = folder_path / "template.tex"
            if not meta_path or not tex_path.exists():
                print(f"  [SKIP] {folder_path.name} - missing metadata.json/Metadata.json or template.tex")
                skipped += 1
                continue

            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                tex = tex_path.read_text(encoding="utf-8")

                slug = meta.get("id") or folder_path.name
                existing = ResumeTemplate.query.filter_by(slug=slug).first()

                if existing:
                    existing.name = meta.get("name", slug)
                    existing.category = meta.get("category", "tech")
                    existing.best_for = meta.get("best_for", "")
                    existing.description = meta.get("description", "")
                    existing.latex_code = tex
                    existing.is_active = True
                    print(f"  [UPDATE] {slug}")
                else:
                    db.session.add(ResumeTemplate(
                        name=meta.get("name", slug),
                        slug=slug,
                        category=meta.get("category", "tech"),
                        best_for=meta.get("best_for", ""),
                        description=meta.get("description", ""),
                        latex_code=tex,
                        is_active=True,
                    ))
                    print(f"  [INSERT] {slug}")

                loaded += 1

            except Exception as exc:
                db.session.rollback()
                print(f"  [ERROR] {folder_path.name}: {exc}")
                errors.append(folder_path.name)

        db.session.commit()

    print(f"\nDone. Loaded: {loaded}, Skipped: {skipped}, Errors: {len(errors)}")
    if errors:
        print(f"Failed: {errors}")


if __name__ == "__main__":
    main()
