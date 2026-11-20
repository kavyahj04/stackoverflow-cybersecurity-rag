import os

import psycopg
from dotenv import load_dotenv

load_dotenv(override=True)

with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    database, user = conn.execute("SELECT current_database(), current_user").fetchone()
    vector = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    tables = [row[0] for row in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")]

print("Connected to database:", database, "as user:", user)
print("pgvector version:", vector[0] if vector else "NOT turned on yet")
print("Tables:", tables or "none yet")
