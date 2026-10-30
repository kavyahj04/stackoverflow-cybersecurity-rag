"""Dense search over the questions collection (title + tags + question body).

One vector per thread, so a hit is a thread directly. Does not load the
cross-encoder or BM25, so it starts fast. Separate from 08_hybrid_search.py,
which keeps the answer-based search.
"""
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

SRC_DIR = Path(__file__).resolve().parent.parent
load_dotenv(SRC_DIR.parent / ".env")

EMBEDDING_MODEL = "text-embedding-3-large"  # must match the model used in 05_embed_qa.py

_client = OpenAI()
_questions = chromadb.PersistentClient(path=str(SRC_DIR / "chroma_db")).get_collection("questions")


def embed_query(text):
    return _client.embeddings.create(input=[text], model=EMBEDDING_MODEL).data[0].embedding


def search_threads(query, top_k=10):
    """Return the top_k thread ids (question ids), best first, with Chroma distances."""
    result = _questions.query(query_embeddings=[embed_query(query)], n_results=top_k)
    return list(zip(result["ids"][0], result["distances"][0]))
