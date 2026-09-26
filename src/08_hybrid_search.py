import importlib
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

SRC_DIR = Path(__file__).resolve().parent

bm25_index = importlib.import_module("07_bm25_index")
fusion = importlib.import_module("10_fusion")
load_index, tokenize = bm25_index.load_index, bm25_index.tokenize

load_dotenv()

client = OpenAI()
chroma_client = chromadb.PersistentClient(path=str(SRC_DIR / "chroma_db"))
answers_collection = chroma_client.get_collection("answers")

bm25, bm25_ids = load_index()

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


def sparse_search(question, top_k=50):
    tokens = tokenize(question)
    scores = bm25.get_scores(tokens)
    ranked = sorted(zip(bm25_ids, scores), key=lambda x: x[1], reverse=True)[:top_k]
    ranked_ids = [answer_id for answer_id, _ in ranked]

    # Pull display text from Chroma (already stores it) instead of
    # re-parsing/re-cleaning the raw XML just to look up a few strings.
    fetched = answers_collection.get(ids=ranked_ids, include=["documents"])
    text_by_id = dict(zip(fetched["ids"], fetched["documents"]))

    hits = []
    for answer_id, score in ranked:
        hits.append({
            "answer_id": answer_id,
            "score": score,
            "text": text_by_id[answer_id],
            "url": answer_url(answer_id),
        })
    return hits


def hybrid_search(question, top_k=50):
    dense_hits =  dense_search(question, top_k)
    sparse_hits = sparse_search(question, top_k)
    return {
        "dense": dense_hits,
        "sparse": sparse_hits,
        "fused" : fusion.reciprocal_rank_fusion(dense_hits, sparse_hits)
    }


def print_hits(label, hits):
    print(f"\n--- {label} ---")
    for rank, hit in enumerate(hits, start=1):
        snippet = hit["text"][:150].replace("\n", " ")
        print(f"{rank}. [{hit['answer_id']}] score={hit['score']:.4f}  {snippet}...")
        print(f"    {hit['url']}")


if __name__ == "__main__":
    test_questions = [
        "BCrypt workfactor for salt"
    ]

    for question in test_questions:
        print(f"\n===== Question: {question} =====")
        results = hybrid_search(question, top_k=50)
        print_hits("Dense (embeddings)", results["dense"])
        print_hits("Sparse (BM25)", results["sparse"])
        print_hits("Fused (RRF)", results["fused"])
