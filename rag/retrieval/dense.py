"""Dense (embedding) search over questions with pgvector's HNSW index.

One vector per thread, so each hit is a thread. Postgres uses the HNSW index
by itself for ORDER BY embedding <=> query, because the index was built with
halfvec_cosine_ops and <=> is cosine distance.
"""
import numpy as np
from openai import OpenAI
from pgvector import HalfVector

from rag.config import get_settings

SEARCH_SQL = """
    SELECT question_id, embedding <=> %(query)s AS distance
    FROM questions
    ORDER BY embedding <=> %(query)s
    LIMIT %(top_k)s
"""


class DenseRetriever:
    def __init__(self, conn, client: OpenAI | None = None):
        self.conn = conn  # from rag.db.connect()
        self.client = client or OpenAI()
        self.model = get_settings().embedding_model

    def embed(self, text: str) -> list[float]:
        return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding

    def search(self, query: str, top_k: int | None = None) -> list[tuple[int, float]]:
        """Top question ids for the query, best first, with cosine distances (0 = same meaning)."""
        top_k = top_k or get_settings().top_k
        # Same 2-byte type as the stored column, so the comparison matches the index
        query_vector = HalfVector(np.array(self.embed(query), dtype=np.float16))
        rows = self.conn.execute(SEARCH_SQL, {"query": query_vector, "top_k": top_k}).fetchall()
        return [(question_id, float(distance)) for question_id, distance in rows]
