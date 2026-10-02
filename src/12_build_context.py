import importlib
import math
import os
import sqlite3
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
THREADS_DB = SRC_DIR / "threads.db"
CONTEXT_OUTPUT_FILE = SRC_DIR / "context_results.txt"

# Quality score used to pick answers inside a thread. Votes and accepted
# dominate; recency is a small tie-breaker so timeless answers survive.
W_VOTES = 1.0       # * log(1 + votes)
W_ACCEPTED = 3.0    # flat bonus for the accepted answer
W_RECENCY = -0.05   # per year of age

MAX_THREADS = 5
MAX_ANSWERS_PER_THREAD = 5


SCHEMA = """
CREATE TABLE questions (
    question_id        TEXT PRIMARY KEY,
    title              TEXT,
    body               TEXT,
    votes              INTEGER,
    creation_date      TEXT,
    accepted_answer_id TEXT
);
CREATE TABLE answers (
    answer_id     TEXT PRIMARY KEY,
    question_id   TEXT NOT NULL REFERENCES questions(question_id),
    body          TEXT,
    votes         INTEGER,
    creation_date TEXT,
    is_accepted   INTEGER
);
CREATE INDEX idx_answers_question ON answers(question_id);
"""


def build_threads_db(path):
    """Build the thread database once from the parsed Q&A (slow: re-runs parsing 03/04)."""
    parsing_answers = importlib.import_module("03_parsing_answers")
    parsing_questions = importlib.import_module("04_parsing_questions")

    # Build in a temp file and rename at the end, so an interrupted build
    # never leaves a half-written database behind.
    tmp_path = f"{path}.tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    conn = sqlite3.connect(tmp_path)
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT INTO questions VALUES (?, ?, ?, ?, ?, ?)",
        [
            (q["question_id"], q["title"], q["body_clean"], int(q["score"] or 0),
             q["creation_date"], q["accepted_answer_id"])
            for q in parsing_questions.question_records
        ],
    )
    conn.executemany(
        "INSERT INTO answers VALUES (?, ?, ?, ?, ?, ?)",
        [
            (a["answer_id"], a["thread_id"], a["answer_body_clean"], int(a["score"] or 0),
             a["creation_date"], int(a["is_accepted"]))
            for a in parsing_answers.answer_records
        ],
    )
    conn.commit()
    conn.close()
    os.replace(tmp_path, path)


class ThreadStore:
    """Read access to threads (a question plus all its answers) in SQLite.

    Only the two lookups build_context needs, so the backing store can be
    swapped (e.g. Postgres) without touching the ranking code.
    """

    def __init__(self, path=THREADS_DB):
        if not Path(path).exists():
            build_threads_db(path)
        self.conn = sqlite3.connect(path)

    def thread_ids_for(self, answer_ids):
        """answer_id -> thread_id (the question the answer belongs to)."""
        answer_ids = list(answer_ids)
        marks = ",".join("?" * len(answer_ids))
        rows = self.conn.execute(
            f"SELECT answer_id, question_id FROM answers WHERE answer_id IN ({marks})",
            answer_ids,
        )
        return dict(rows)

    def get_threads(self, thread_ids):
        """thread_id -> {"question": {...}, "answers": [{...}]}"""
        thread_ids = list(thread_ids)
        marks = ",".join("?" * len(thread_ids))
        threads = {}
        for qid, title, body, votes, created, accepted_id in self.conn.execute(
            "SELECT question_id, title, body, votes, creation_date, accepted_answer_id "
            f"FROM questions WHERE question_id IN ({marks})",
            thread_ids,
        ):
            threads[qid] = {
                "question": {
                    "title": title,
                    "body": body,
                    "votes": votes,
                    "creation_date": created,
                    "accepted_answer_id": accepted_id,
                },
                "answers": [],
            }
        for aid, qid, body, votes, created, is_accepted in self.conn.execute(
            "SELECT answer_id, question_id, body, votes, creation_date, is_accepted "
            f"FROM answers WHERE question_id IN ({marks})",
            thread_ids,
        ):
            threads[qid]["answers"].append({
                "answer_id": aid,
                "body": body,
                "votes": votes,
                "creation_date": created,
                "is_accepted": bool(is_accepted),
            })
        return threads


def age_years(creation_date):
    created = datetime.fromisoformat(creation_date)
    return max((datetime.now() - created).days / 365.25, 0)


def build_context(question, hits, store, cross_encoder,
                  max_threads=MAX_THREADS, max_answers=MAX_ANSWERS_PER_THREAD):
    """Expand retrieved answers to whole threads and pick what to send to the LLM."""
    matched_ids = {hit["answer_id"] for hit in hits}

    # 1. Threads behind the retrieved answers, then load them from the store.
    answer_to_thread = store.thread_ids_for(matched_ids)
    thread_ids = []
    for hit in hits:
        thread_id = answer_to_thread[hit["answer_id"]]
        if thread_id not in thread_ids:
            thread_ids.append(thread_id)
    threads = store.get_threads(thread_ids)

    # 2. Inside each thread, pick the best answers by quality (accepted + votes +
    #    a little recency). Downvoted answers are dropped unless accepted.
    #    The accepted answer is always among the picks.
    candidates = []
    for thread_id in thread_ids:
        thread = threads[thread_id]
        answers = []
        for ans in thread["answers"]:
            if ans["votes"] < 0 and not ans["is_accepted"]:
                continue
            answers.append({
                **ans,
                "thread_id": thread_id,
                "matched": ans["answer_id"] in matched_ids,
                "quality": (
                    W_VOTES * math.log1p(max(ans["votes"], 0))
                    + (W_ACCEPTED if ans["is_accepted"] else 0)
                    + W_RECENCY * age_years(ans["creation_date"])
                ),
                "text": f"{thread['question']['title']} | {ans['body']}",
            })
        answers.sort(key=lambda a: a["quality"], reverse=True)
        candidates.extend(answers[:max_answers])

    # 3. Cross-encoder checks each picked answer against the user's question.
    #    rerank() returns copies with "score" set to the cross-encoder logit.
    scored = cross_encoder.rerank(question, candidates)
    for ans in scored:
        ans["relevance"] = ans.pop("score")

    # 4. Group by thread. Nothing is dropped here: relevance only orders results.
    by_thread = {}
    for ans in scored:
        by_thread.setdefault(ans["thread_id"], []).append(ans)

    # 5. Best threads first (by their most relevant answer); inside a thread,
    #    higher-quality answers first.
    blocks = []
    for thread_id, answers in by_thread.items():
        answers.sort(key=lambda a: a["quality"], reverse=True)
        blocks.append({
            "thread_id": thread_id,
            "question": threads[thread_id]["question"],
            "answers": answers,
            "thread_score": max(a["relevance"] for a in answers),
        })
    blocks.sort(key=lambda b: b["thread_score"], reverse=True)
    return blocks[:max_threads]


def format_context(blocks):
    """Prompt-ready text: question, then its answers labelled with accepted/votes/year."""
    parts = []
    for block in blocks:
        q = block["question"]
        parts.append(f"### Question: {q['title']}\n{q['body']}")
        for ans in block["answers"]:
            flags = (["ACCEPTED"] if ans["is_accepted"] else []) + [
                f"{ans['votes']:+d} votes", ans["creation_date"][:4]
            ]
            parts.append(f"[Answer {ans['answer_id']} | {', '.join(flags)}]\n{ans['body']}")
    return "\n\n".join(parts)


def write_context_report(path, questions, search, store, cross_encoder):
    """Compact per-question report so we can eyeball which answers get picked."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Cross-encoder: {cross_encoder.MODEL_NAME}\n")
        f.write("* = answer was in the retrieved top 10, A = accepted\n")
        for question in questions:
            hits = search(question, top_k=50)["final"]
            blocks = build_context(question, hits, store, cross_encoder)
            f.write(f"\n===== {question} =====\n")
            for block in blocks:
                f.write(f"  [{block['thread_score']:5.2f}] {block['question']['title'][:90]}\n")
                for ans in block["answers"]:
                    mark = ("*" if ans["matched"] else " ") + ("A" if ans["is_accepted"] else " ")
                    f.write(f"      {mark} {ans['answer_id']:>7} votes={ans['votes']:>4} "
                            f"{ans['creation_date'][:4]} quality={ans['quality']:5.2f} "
                            f"rel={ans['relevance']:5.2f}\n")
            print(f"done: {question}")


if __name__ == "__main__":
    hybrid = importlib.import_module("08_hybrid_search")
    store = ThreadStore()
    write_context_report(
        CONTEXT_OUTPUT_FILE,
        hybrid.TEST_QUESTIONS,
        hybrid.hybrid_search,
        store,
        hybrid.cross_encoder,
    )
    print(f"\nWrote {CONTEXT_OUTPUT_FILE}")
