import psycopg2

SUPABASE_URL = "postgresql://postgres.jqpeqwzeustootixllmd:NirvanaXmahi%4030@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres"

conn = psycopg2.connect(SUPABASE_URL)
cur = conn.cursor()
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
tables = [r[0] for r in cur.fetchall()]
print("Tables in Supabase:", tables)
conn.close()
