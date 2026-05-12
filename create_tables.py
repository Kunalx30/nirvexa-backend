import psycopg2

SUPABASE_URL = "postgresql://postgres.jqpeqwzeustootixllmd:NirvanaXmahi%4030@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres"

conn = psycopg2.connect(SUPABASE_URL)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS chat_history (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    model_used VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS interview_sessions (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(200),
    mode VARCHAR(50),
    difficulty VARCHAR(50),
    total_score FLOAT,
    avg_wpm FLOAT,
    filler_word_count INTEGER,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS interview_responses (
    id VARCHAR PRIMARY KEY,
    session_id VARCHAR REFERENCES interview_sessions(id) ON DELETE CASCADE,
    question_number INTEGER,
    question_text TEXT,
    transcript TEXT,
    content_score FLOAT,
    keyword_score FLOAT,
    grammar_score FLOAT,
    confidence_score FLOAT,
    overall_score FLOAT,
    feedback TEXT,
    suggested_answer TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
""")

conn.commit()
conn.close()
print("Tables created successfully!")
