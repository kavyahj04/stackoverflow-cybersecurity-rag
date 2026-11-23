"""Build the exact-term test set: queries that contain a CVE id.

For every CVE id in a question title, the correct threads are all questions
whose title contains that id. Each CVE gets two queries: the bare id (how
researchers paste it) and the id inside a short generic question. The
templates use no words from the title, so no search method gets a free hint.

CVEs are split into dev (use while building) and test (score once, at the
end) by a hash of the id, so the split is the same on every run.

Usage (from rag-dev):  python -m rag.eval.build_exact_set
"""
import hashlib
import json
import re
from collections import defaultdict

from rag.config import ROOT_DIR, SRC_DIR
from rag.db import connect

CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
TEMPLATES = [
    "What is {term}?",
    "How do I fix {term}?",
    "Is {term} exploitable?",
    "Am I affected by {term}?",
]
TITLES_SQL = """
    SELECT question_id, title FROM questions
    WHERE title ~* 'CVE-[0-9]{4}-[0-9]{4,7}'
    ORDER BY question_id
"""


def split_of(term: str) -> str:
    """dev or test, decided by the id's hash, so it never changes between runs."""
    return "dev" if int(hashlib.sha256(term.encode()).hexdigest(), 16) % 2 == 0 else "test"


def main():
    conn = connect()

    # CVE id -> every question whose title contains it (oldest first)
    threads = defaultdict(list)
    for qid, title in conn.execute(TITLES_SQL):
        for term in sorted({m.upper() for m in CVE_PATTERN.findall(title)}):
            threads[term].append(qid)

    all_ids = [qid for ids in threads.values() for qid in ids]
    accepted = dict(conn.execute(
        "SELECT question_id, accepted_answer_id FROM questions WHERE question_id = ANY(%s)", (all_ids,)
    ).fetchall())

    cases = {"dev": [], "test": []}
    for i, (term, ids) in enumerate(sorted(threads.items())):
        gold, *others = ids
        question = TEMPLATES[i % len(TEMPLATES)].format(term=term)
        for style, query in (("exact_bare", term), ("exact_question", question)):
            cases[split_of(term)].append({
                "query": query,
                "style": style,
                "bucket": "cve",
                "term": term,
                "gold_thread_id": gold,
                "also_correct_threads": others,
                "gold_answer_ids": [],
                "accepted_answer_id": accepted[gold],
            })

    for name, rows in cases.items():
        path = SRC_DIR / f"test_set_exact_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"{name}: {len(rows)} queries, {len({r['term'] for r in rows})} CVE ids -> {path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
