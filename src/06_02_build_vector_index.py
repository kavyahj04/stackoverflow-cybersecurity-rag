import os
import time

import psycopg
from dotenv import load_dotenv

load_dotenv(override=True)

INDEX_SQL = """
    CREATE INDEX questions_embedding_hnsw_idx
    ON questions USING hnsw (embedding halfvec_cosine_ops)
"""

# One transaction: if building the new index fails, the old index (if any) stays
with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    # Supabase cancels commands after 2 minutes; building this index takes longer
    conn.execute("SET statement_timeout = 0")

    start = time.time()
    conn.execute("DROP INDEX IF EXISTS questions_embedding_hnsw_idx")
    conn.execute(INDEX_SQL)
    print(f"Index built in {time.time() - start:.0f}s", flush=True)

    # Refresh Postgres's statistics so it plans queries well
    conn.execute("ANALYZE questions")
    conn.execute("ANALYZE answers")

    size = conn.execute(
        "SELECT pg_size_pretty(pg_relation_size('questions_embedding_hnsw_idx'))"
    ).fetchone()[0]
    print(f"Index size: {size}")
