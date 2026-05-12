import psycopg2
import json

SUPABASE_URL = "postgresql://postgres.jqpeqwzeustootixllmd:NirvanaXmahi%4030@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres"

with open("nirvexa_backup.json", "r") as f:
    backup = json.load(f)

conn = psycopg2.connect(SUPABASE_URL)
cur = conn.cursor()

# Add ALL missing interview_sessions columns
for col in ["grammar_score FLOAT", "keyword_score FLOAT", "communication_score FLOAT", 
            "clarity_score FLOAT", "relevance_score FLOAT", "structure_score FLOAT",
            "feedback TEXT", "questions_asked INTEGER", "duration_seconds INTEGER"]:
    cur.execute(f"ALTER TABLE interview_sessions ADD COLUMN IF NOT EXISTS {col};")
conn.commit()
print("Schema fixed!")

# Find all ARRAY columns per table
def get_array_cols(table):
    cur.execute("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name=%s AND table_schema='public' AND data_type='ARRAY'
    """, (table,))
    return {r[0] for r in cur.fetchall()}

table_order = ["users", "interview_sessions", "interview_responses", "jobs", "saved_jobs", "job_alerts", "news_cache", "resume_analyses"]

for table in table_order:
    rows = backup.get(table, [])
    if not rows:
        print(f"  {table}: empty, skipping")
        continue

    # Get DB columns and array columns
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name=%s AND table_schema='public'", (table,))
    db_cols = {r[0] for r in cur.fetchall()}
    array_cols = get_array_cols(table)

    cols = [c for c in rows[0].keys() if c in db_cols]
    placeholders = ",".join(["%s"] * len(cols))
    col_names = ",".join(cols)

    inserted = 0
    errors = []
    for row in rows:
        try:
            values = []
            for c in cols:
                v = row[c]
                if c in array_cols:
                    # Convert to proper PostgreSQL array
                    if v is None:
                        v = []
                    elif isinstance(v, str):
                        try:
                            v = json.loads(v)
                        except:
                            v = []
                    # v is now a Python list, psycopg2 sends it as proper PG array
                elif isinstance(v, (dict, list)):
                    v = json.dumps(v)
                values.append(v)
            cur.execute(f"INSERT INTO {table} ({col_names}) VALUES ({placeholders}) ON CONFLICT DO NOTHING", values)
            inserted += 1
        except Exception as e:
            errors.append(str(e))
            conn.rollback()

    conn.commit()
    print(f"  {table}: {inserted} inserted, {len(errors)} errors")
    if errors:
        print(f"    First error: {errors[0]}")

conn.close()
print("\nDone!")
