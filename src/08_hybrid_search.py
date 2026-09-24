import importlib
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

SRC_DIR = Path(__file__).resolve().parent

bm25_index = importlib.import_module("07_bm25_index")
load_index, tokenize = bm25_index.load_index, bm25_index.tokenize
parsing_answers = importlib.import_module("03_parsing_answers")
answer_records = parsing_answers.answer_records

load_dotenv()

client = OpenAI()
chroma_client = chromadb.PersistentClient(path=str(SRC_DIR / "chroma_db"))
answers_collection = chroma_client.get_collection("answers")

bm25, bm25_ids = load_index()
records_by_id = {r["answer_id"]: r for r in answer_records}

SITE_BASE_URL = "https://security.stackexchange.com"


def answer_url(answer_id):
    return f"{SITE_BASE_URL}/a/{answer_id}"


def embed_query(text):
    response = client.embeddings.create(input=[text], model="text-embedding-3-small")
    return response.data[0].embedding


def dense_search(question, top_k=5):
    query_embedding = embed_query(question)
    results = answers_collection.query(query_embeddings=[query_embedding], n_results=top_k)
    hits = []
    for answer_id, distance, document in zip(
        results["ids"][0], results["distances"][0], results["documents"][0]
    ):
        hits.append({"answer_id": answer_id, "score": distance, "text": document, "url": answer_url(answer_id)})
    return hits


def sparse_search(question, top_k=5):
    tokens = tokenize(question)
    scores = bm25.get_scores(tokens)
    ranked = sorted(zip(bm25_ids, scores), key=lambda x: x[1], reverse=True)[:top_k]
    hits = []
    for answer_id, score in ranked:
        hits.append({
            "answer_id": answer_id,
            "score": score,
            "text": records_by_id[answer_id]["question_info"],
            "url": answer_url(answer_id),
        })
    return hits


def hybrid_search(question, top_k=5):
    return {
        "dense": dense_search(question, top_k),
        "sparse": sparse_search(question, top_k),
    }


def print_hits(label, hits):
    print(f"\n--- {label} ---")
    for rank, hit in enumerate(hits, start=1):
        snippet = hit["text"][:150].replace("\n", " ")
        print(f"{rank}. [{hit['answer_id']}] score={hit['score']:.4f}  {snippet}...")
        print(f"    {hit['url']}")


if __name__ == "__main__":
    test_questions = [
        "How do I prevent SQL injection attacks?",
        "What is the difference between symmetric and asymmetric encryption?",
        "How does a buffer overflow attack work?",
        "What are best practices for storing passwords securely?",
        "How can I detect if my website is vulnerable to XSS?",
    ]

    for question in test_questions:
        print(f"\n===== Question: {question} =====")
        results = hybrid_search(question, top_k=5)
        print_hits("Dense (embeddings)", results["dense"])
        print_hits("Sparse (BM25)", results["sparse"])
