import psycopg2
import json
from datetime import datetime, date

DB_URL = "postgresql://nirvexa_db_user:QvyBsdSM2SJgUBxeoScSfgmfm8iDGote@dpg-d7du097lk1mc73f0v5hg-a.singapore-postgres.render.com/nirvexa_db"

def serialize(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return str(obj)

conn = psycopg2.connect(DB_URL)
cur = conn.cursor()

cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
tables = [r[0] for r in cur.fetchall()]
print("Tables found:", tables)

backup = {}

for table in tables:
    cur.execute(f"SELECT * FROM {table}")
    rows = cur.fetchall()
    col_names = [desc[0] for desc in cur.description]
    backup[table] = [dict(zip(col_names, row)) for row in rows]
    print(f"  {table}: {len(rows)} rows exported")

conn.close()

with open("nirvexa_backup.json", "w") as f:
    json.dump(backup, f, default=serialize, indent=2)

print("Done! Saved to nirvexa_backup.json")
