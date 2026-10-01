import importlib
import math
import pickle
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
THREADS_CACHE = SRC_DIR / "threads_index.pkl"
CONTEXT_OUTPUT_FILE = SRC_DIR / "context_results.txt"

# Ranking weights applied on top of the cross-encoder logit (roughly -10..+10).
W_VOTES = 0.5       # * log(1 + votes)
W_ACCEPTED = 1.0    # flat bonus for the accepted answer
W_RECENCY = -0.1    # per year of age: mild, so timeless answers survive

MAX_THREADS = 5
MAX_ANSWERS_PER_THREAD = 3


def build_threads_index():
    """Return (threads, answer_to_thread) built once from the parsed Q&A.

    threads: thread_id -> {"question": {...}, "answers": [{...}]}
    """
    parsing_answers = importlib.import_module("03_parsing_answers")
    parsing_questions = importlib.import_module("04_parsing_questions")

    threads = {}
    for q in parsing_questions.question_records:
        threads[q["question_id"]] = {
            "question": {
                "title": q["title"],
                "body": q["body_clean"],
                "votes": int(q["score"] or 0),
                "creation_date": q["creation_date"],
                "accepted_answer_id": q["accepted_answer_id"],
            },
            "answers": [],
        }
    answer_to_thread = {}
    for a in parsing_answers.answer_records:
        threads[a["thread_id"]]["answers"].append({
            "answer_id": a["answer_id"],
            "body": a["answer_body_clean"],
            "votes": int(a["score"] or 0),
            "creation_date": a["creation_date"],
            "is_accepted": a["is_accepted"],
        })
        answer_to_thread[a["answer_id"]] = a["thread_id"]
    return threads, answer_to_thread


def load_threads():
    if THREADS_CACHE.exists():
        with open(THREADS_CACHE, "rb") as f:
            return pickle.load(f)
    index = build_threads_index()
    with open(THREADS_CACHE, "wb") as f:
        pickle.dump(index, f)
    return index


def age_years(creation_date):
    created = datetime.fromisoformat(creation_date)
    return max((datetime.now() - created).days / 365.25, 0)


def build_context(question, hits, threads, answer_to_thread, cross_encoder,
                  max_threads=MAX_THREADS, max_answers=MAX_ANSWERS_PER_THREAD):
    """Expand retrieved answers to whole threads and pick what to send to the LLM."""
    matched_ids = {hit["answer_id"] for hit in hits}

    # 1. Threads behind the retrieved answers.
    thread_ids = []
    for hit in hits:
        thread_id = answer_to_thread[hit["answer_id"]]
        if thread_id not in thread_ids:
            thread_ids.append(thread_id)

    # 2. Every answer in those threads (drop downvoted ones unless accepted).
    candidates = []
    for thread_id in thread_ids:
        thread = threads[thread_id]
        for ans in thread["answers"]:
            if ans["votes"] < 0 and not ans["is_accepted"]:
                continue
            candidates.append({
                **ans,
                "thread_id": thread_id,
                "matched": ans["answer_id"] in matched_ids,
                "text": f"{thread['question']['title']} | {ans['body']}",
            })

    # 3. Relevance to the user's question, then votes / accepted / recency.
    #    rerank() returns copies with "score" set to the cross-encoder logit.
    by_thread = {}
    for ans in cross_encoder.rerank(question, candidates):
        ans["relevance"] = ans.pop("score")
        ans["rank_score"] = (
            ans["relevance"]
            + W_VOTES * math.log1p(max(ans["votes"], 0))
            + (W_ACCEPTED if ans["is_accepted"] else 0)
            + W_RECENCY * age_years(ans["creation_date"])
        )
        by_thread.setdefault(ans["thread_id"], []).append(ans)

    # 4. Best answers per thread (accepted answer is always kept), best threads first.
    blocks = []
    for thread_id, answers in by_thread.items():
        answers.sort(key=lambda a: a["rank_score"], reverse=True)
        chosen = answers[:max_answers]
        accepted = next((a for a in answers if a["is_accepted"]), None)
        if accepted and accepted not in chosen:
            chosen = chosen[:-1] + [accepted]
        blocks.append({
            "thread_id": thread_id,
            "question": threads[thread_id]["question"],
            "answers": chosen,
            "thread_score": answers[0]["rank_score"],
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


def write_context_report(path, questions, search, threads, answer_to_thread, cross_encoder):
    """Compact per-question report so we can eyeball which answers get picked."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Cross-encoder: {cross_encoder.MODEL_NAME}\n")
        f.write("* = answer was in the retrieved top 10, A = accepted\n")
        for question in questions:
            hits = search(question, top_k=50)["final"]
            blocks = build_context(question, hits, threads, answer_to_thread, cross_encoder)
            f.write(f"\n===== {question} =====\n")
            for block in blocks:
                f.write(f"  [{block['thread_score']:5.2f}] {block['question']['title'][:90]}\n")
                for ans in block["answers"]:
                    mark = ("*" if ans["matched"] else " ") + ("A" if ans["is_accepted"] else " ")
                    f.write(f"      {mark} {ans['answer_id']:>7} votes={ans['votes']:>4} "
                            f"{ans['creation_date'][:4]} rank={ans['rank_score']:5.2f} "
                            f"rel={ans['relevance']:5.2f}\n")
            print(f"done: {question}")


if __name__ == "__main__":
    hybrid = importlib.import_module("08_hybrid_search")
    threads, answer_to_thread = load_threads()
    write_context_report(
        CONTEXT_OUTPUT_FILE,
        hybrid.TEST_QUESTIONS,
        hybrid.hybrid_search,
        threads,
        answer_to_thread,
        hybrid.cross_encoder,
    )
    print(f"\nWrote {CONTEXT_OUTPUT_FILE}")
