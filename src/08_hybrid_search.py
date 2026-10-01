import importlib
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

SRC_DIR = Path(__file__).resolve().parent

bm25_index = importlib.import_module("07_bm25_index")
fusion = importlib.import_module("10_fusion")
load_index, tokenize = bm25_index.load_index, bm25_index.tokenize

cross_encoder = importlib.import_module("11_cross_encoder")

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

def merge_with_floor(dense_floor, reranked, final_size=10):
    final = list(dense_floor)
    seen_ids = {hit["answer_id"] for hit in final}
    for hit in reranked:
        if len(final) >= final_size:
            break
        if hit["answer_id"] not in seen_ids:
            final.append(hit)
            seen_ids.add(hit["answer_id"])
    return final



def hybrid_search(question, top_k=50, final_size=10):
    dense_hits =  dense_search(question, top_k)
    sparse_hits = sparse_search(question, top_k)

    dense_floor = dense_hits[:5]

    # Candidate pool: dense ranks 6-50 + all sparse hits, deduped by answer_id
    # so each answer is scored by the cross-encoder only once.
    pool = {}
    for hit in dense_hits[5:] + sparse_hits:
        pool.setdefault(hit["answer_id"], hit)
    reranked = cross_encoder.rerank(question, list(pool.values()))

    final = merge_with_floor(dense_floor, reranked, final_size)
    return {
        "dense": dense_hits,
        "sparse": sparse_hits,
        "reranked": reranked,
        "final" : final,
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
        print_hits("Final (dense floor + cross-encoder fill)", results["final"])
