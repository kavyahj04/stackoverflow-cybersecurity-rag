"""Dense search over questions in Postgres (pgvector HNSW index).

One vector per thread, so a hit is a thread directly. Postgres decides by
itself to use the HNSW index for ORDER BY embedding <=> query.
"""
import os
from pathlib import Path
import numpy as np
import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from pgvector import HalfVector
from pgvector.psycopg import register_vector

SRC_DIR = Path(__file__).resolve().parent.parent
load_dotenv(SRC_DIR.parent / ".env")

EMBEDDING_MODEL = "text-embedding-3-large"  # must match the model used in 05_embed_qa.py

_client = OpenAI()
_conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
register_vector(_conn)

SEARCH_SQL = """
    SELECT question_id, embedding <=> %(query)s AS distance
    FROM questions
    ORDER BY embedding <=> %(query)s
    LIMIT %(top_k)s
"""

def embed_query(text):
    return _client.embeddings.create(input=[text], model=EMBEDDING_MODEL).data[0].embedding


def search_threads(query, top_k=10):
    """Return the top_k thread ids (question ids), best first, with cosine distances."""
    query_vector = HalfVector(np.array(embed_query(query), dtype=np.float16))
    rows = _conn.execute(SEARCH_SQL, {"query": query_vector, "top_k": top_k}).fetchall()
    return [(str(question_id), float(distance)) for question_id, distance in rows]