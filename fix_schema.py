import psycopg2
import json

SUPABASE_URL = "postgresql://postgres.jqpeqwzeustootixllmd:NirvanaXmahi%4030@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres"

with open("nirvexa_backup.json", "r") as f:
    backup = json.load(f)

conn = psycopg2.connect(SUPABASE_URL)
cur = conn.cursor()

# Add ALL possibly missing columns to interview_sessions
extra_cols = [
    "confidence_score FLOAT",
    "communication_score FLOAT",
    "clarity_score FLOAT",
    "relevance_score FLOAT",
    "structure_score FLOAT",
    "feedback TEXT",
    "questions_asked INTEGER",
    "duration_seconds INTEGER"
]
for col in extra_cols:
    try:
        cur.execute(f"ALTER TABLE interview_sessions ADD COLUMN IF NOT EXISTS {col};")
    except:
        pass
conn.commit()
print("Schema fixed!")

# Check what columns users table actually has
cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='users' ORDER BY ordinal_position")
user_cols = cur.fetchall()
print("Users columns:", user_cols)

conn.close()
