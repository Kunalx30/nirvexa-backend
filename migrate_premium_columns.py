from app import create_app
from app.database.db import db
from sqlalchemy import text

app = create_app()

with app.app_context():
    with db.engine.connect() as conn:
        # PostgreSQL way to check existing columns
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'users'
        """))
        existing = [row[0] for row in result.fetchall()]

        if "is_premium" not in existing:
            conn.execute(text("ALTER TABLE users ADD COLUMN is_premium BOOLEAN DEFAULT FALSE"))
            print("Added: is_premium")
        else:
            print("Skipped: is_premium already exists")

        if "premium_plan" not in existing:
            conn.execute(text("ALTER TABLE users ADD COLUMN premium_plan VARCHAR(50)"))
            print("Added: premium_plan")
        else:
            print("Skipped: premium_plan already exists")

        if "premium_expiry" not in existing:
            conn.execute(text("ALTER TABLE users ADD COLUMN premium_expiry TIMESTAMP"))
            print("Added: premium_expiry")
        else:
            print("Skipped: premium_expiry already exists")

        conn.commit()
        print("Migration complete.")