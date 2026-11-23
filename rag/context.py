"""Turn retrieved threads into the context the LLM reads.

Threads keep the order search returned them (re-ordering with a cross-encoder
lowered recall, so it is not used). Inside each thread, answers are picked by
a quality score: votes and the accepted mark dominate, age is a small nudge.
No model is called here.
"""
import math
from datetime import datetime

from rag.config import get_settings

W_VOTES = 1.0       # * log(1 + score)
W_ACCEPTED = 3.0    # flat bonus for the accepted answer
W_RECENCY = -0.05   # per year of age


def age_years(created: datetime, now: datetime) -> float:
    return max((now - created).days / 365.25, 0)


def quality(answer: dict, now: datetime) -> float:
    return (W_VOTES * math.log1p(max(answer["score"], 0))
            + (W_ACCEPTED if answer["is_accepted"] else 0)
            + W_RECENCY * age_years(answer["creation_date"], now))


def pick_answers(answers: list[dict], max_answers: int, now: datetime) -> list[dict]:
    """Best answers by quality. Downvoted ones are dropped unless accepted; the accepted one is always kept."""
    eligible = [a for a in answers if a["score"] >= 0 or a["is_accepted"]]
    eligible.sort(key=lambda a: quality(a, now), reverse=True)
    chosen = eligible[:max_answers]
    accepted = next((a for a in eligible if a["is_accepted"]), None)
    if accepted and accepted not in chosen:
        chosen = chosen[:-1] + [accepted]
    return chosen


def build_context(question_ids: list[int], store, max_threads: int | None = None,
                  max_answers: int | None = None, now: datetime | None = None) -> list[dict]:
    """question_ids: retrieved threads, best first. Returns one block per thread, in that order."""
    settings = get_settings()
    max_threads = max_threads or settings.top_k
    max_answers = max_answers or settings.max_answers_per_thread
    now = now or datetime.now()

    question_ids = question_ids[:max_threads]
    threads = store.get_threads(question_ids)
    blocks = []
    for rank, qid in enumerate(question_ids, 1):
        thread = threads.get(qid)
        if thread is None:
            continue
        blocks.append({
            "question_id": qid,
            "rank": rank,
            "question": thread["question"],
            "answers": pick_answers(thread["answers"], max_answers, now),
        })
    return blocks


def format_context(blocks: list[dict]) -> str:
    """Prompt-ready text: each question, then its answers labelled with accepted/score/year."""
    parts = []
    for block in blocks:
        q = block["question"]
        parts.append(f"### Question: {q['title']}\n{q['body']}")
        for ans in block["answers"]:
            flags = (["ACCEPTED"] if ans["is_accepted"] else []) + [
                f"{ans['score']:+d} votes", str(ans["creation_date"].year)
            ]
            parts.append(f"[Answer {ans['answer_id']} | {', '.join(flags)}]\n{ans['body']}")
    return "\n\n".join(parts)
