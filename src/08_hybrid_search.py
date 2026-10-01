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


OUTPUT_FILE = SRC_DIR / "hybrid_search_results.txt"

# Mix of paraphrases (no keyword overlap), ambiguous terms, misconceptions,
# and vague/natural-language phrasing, to stress dense vs sparse vs reranker.
TEST_QUESTIONS = [
    "BCrypt workfactor for salt",
    "is it ok to store passwords encrypted instead of hashed",
    "why does my site break when I put a script tag in the comment box",
    "someone can log in as another user by changing a number in the URL",
    "do I need https if my site has no login",
    "is a longer password always better than a complex one",
    "can hackers see what I do on public wifi",
    "how does a website know I'm the same person between page loads",
    "is it safe to let users upload profile pictures",
    "my api key got pushed to github, what now",
    "what is the difference between encoding, encryption and hashing",
    "can a vpn make me anonymous",
    "why shouldn't I roll my own crypto",
    "is md5 still ok for checksums",
    "how do I stop bots from trying thousands of passwords",
    "what does a salt do if it is stored next to the hash",
    "two factor codes by sms vs authenticator app",
    "can someone steal my login cookie",
    "should passwords expire every 90 days",
    "is it a problem if my certificate is self signed",
    "database query built by string concatenation",
    "how to safely delete data from an ssd",
    "can a pdf file contain a virus",
    "is open source software more secure than closed source",
    "why do websites block pasting into password fields",
    "what happens if two users have the same password",
    "can I trust a password manager with all my passwords",
    "difference between authentication and authorization",
    "is rot13 or base64 a form of security",
    "how long would it take to crack an 8 character password",
]


def write_results(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Cross-encoder: {cross_encoder.MODEL_NAME}\n")
        f.write("Final = dense top 5 (score = distance, lower is better) + "
                "cross-encoder fill (score = logit, higher is better)\n")
        for question in TEST_QUESTIONS:
            results = hybrid_search(question, top_k=50)
            f.write(f"\n===== Question: {question} =====\n")
            for rank, hit in enumerate(results["final"], start=1):
                snippet = hit["text"][:150].replace("\n", " ")
                f.write(f"{rank}. [{hit['answer_id']}] score={hit['score']:.4f}  {snippet}...\n")
                f.write(f"    {hit['url']}\n")
            print(f"done: {question}")


if __name__ == "__main__":
    write_results(OUTPUT_FILE)
    print(f"\nWrote results to {OUTPUT_FILE}")
