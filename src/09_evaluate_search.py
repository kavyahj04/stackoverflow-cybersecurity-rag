import importlib
import json
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
hybrid_search_module = importlib.import_module("08_hybrid_search")
hybrid_search = hybrid_search_module.hybrid_search

JSON_OUTPUT = SRC_DIR / "search_results.json"
TEXT_OUTPUT = SRC_DIR / "search_results.txt"

# Original 5 questions — direct rephrasings of existing SO question titles
ORIGINAL_QUESTIONS = [
    "How do I prevent SQL injection attacks?",
    "What is the difference between symmetric and asymmetric encryption?",
    "How does a buffer overflow attack work?",
    "What are best practices for storing passwords securely?",
    "How can I detect if my website is vulnerable to XSS?",
]

# 10 medium-to-hard multi-hop questions — each requires connecting 2+ concepts
# rather than matching a single existing question title almost verbatim.
MULTIHOP_QUESTIONS = [
    "If I'm using bcrypt to hash passwords, do I still need to add my own random salt, or does bcrypt handle that internally?",
    "Why would enabling HTTPS everywhere still not prevent a man-in-the-middle attack if the certificate authority itself is compromised?",
    "Can a properly configured Content Security Policy alone stop stored XSS if the attacker controls a subdomain via DNS misconfiguration?",
    "If a web application uses parameterized queries everywhere except one debug endpoint, is it still vulnerable to SQL injection, and how would an attacker find that endpoint?",
    "How does the use of asymmetric encryption in a TLS handshake relate to the symmetric encryption used for the rest of the session?",
    "If an attacker can trigger a buffer overflow but the system has ASLR and stack canaries enabled, what techniques could still allow code execution?",
    "Does two-factor authentication protect against a SIM-swapping attack, or does it just shift the vulnerability elsewhere?",
    "Why might rate-limiting login attempts fail to stop a credential-stuffing attack that uses a large botnet?",
    "How does the CIA triad concept apply differently when evaluating an offline air-gapped system versus a cloud-hosted one?",
    "If a company hashes passwords with SHA-256 without salting, what specific attack becomes feasible, and why wouldn't adding a pepper alone fully fix it?",
]

ALL_QUESTIONS = [
    {"question": q, "category": "original"} for q in ORIGINAL_QUESTIONS
] + [
    {"question": q, "category": "multihop"} for q in MULTIHOP_QUESTIONS
]


def run_and_save(top_k=5):
    all_results = []
    text_lines = []

    for item in ALL_QUESTIONS:
        question = item["question"]
        category = item["category"]
        print(f"Running [{category}] {question}")
        results = hybrid_search(question, top_k=top_k)

        all_results.append({
            "question": question,
            "category": category,
            "dense": results["dense"],
            "sparse": results["sparse"],
        })

        text_lines.append(f"\n===== [{category.upper()}] {question} =====")
        text_lines.append("--- Dense (embeddings, lower score = more similar) ---")
        for rank, hit in enumerate(results["dense"], start=1):
            snippet = hit["text"][:150].replace("\n", " ")
            text_lines.append(f"{rank}. [{hit['answer_id']}] score={hit['score']:.4f}  {snippet}...")
        text_lines.append("--- Sparse (BM25, higher score = more similar) ---")
        for rank, hit in enumerate(results["sparse"], start=1):
            snippet = hit["text"][:150].replace("\n", " ")
            text_lines.append(f"{rank}. [{hit['answer_id']}] score={hit['score']:.4f}  {snippet}...")

    with open(JSON_OUTPUT, "w") as f:
        json.dump(all_results, f, indent=2)

    with open(TEXT_OUTPUT, "w") as f:
        f.write("\n".join(text_lines))

    print(f"\nSaved {len(all_results)} question results to:")
    print(f"  {JSON_OUTPUT}")
    print(f"  {TEXT_OUTPUT}")


if __name__ == "__main__":
    run_and_save()
