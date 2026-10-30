"""Turn retrieved threads into the context for the LLM.

Threads keep the order the search returned them (the cross-encoder ordering
used in 12_build_context.py lowered recall, so it is not used here). Inside
each thread, answers are picked by the same quality score as 12_build_context:
votes and accepted dominate, recency is a small nudge. No model is called.
"""
import importlib
import math
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
ctx = importlib.import_module("12_build_context")

MAX_THREADS = 10
MAX_ANSWERS_PER_THREAD = 3


def quality(answer):
    return (ctx.W_VOTES * math.log1p(max(answer["votes"], 0))
            + (ctx.W_ACCEPTED if answer["is_accepted"] else 0)
            + ctx.W_RECENCY * ctx.age_years(answer["creation_date"]))


def pick_answers(answers, max_answers):
    """Best answers by quality; downvoted ones dropped unless accepted; accepted always kept."""
    eligible = [a for a in answers if a["votes"] >= 0 or a["is_accepted"]]
    eligible.sort(key=quality, reverse=True)
    chosen = eligible[:max_answers]
    accepted = next((a for a in eligible if a["is_accepted"]), None)
    if accepted and accepted not in chosen:
        chosen = chosen[:-1] + [accepted]
    return chosen


def build_context(thread_ids, store, max_threads=MAX_THREADS, max_answers=MAX_ANSWERS_PER_THREAD):
    """thread_ids: retrieved threads, best first. Returns blocks in that order."""
    thread_ids = thread_ids[:max_threads]
    threads = store.get_threads(thread_ids)
    blocks = []
    for rank, thread_id in enumerate(thread_ids, 1):
        thread = threads.get(thread_id)
        if thread is None:
            continue
        blocks.append({
            "thread_id": thread_id,
            "rank": rank,
            "question": thread["question"],
            "answers": pick_answers(thread["answers"], max_answers),
        })
    return blocks


format_context = ctx.format_context  # same prompt text format as before
