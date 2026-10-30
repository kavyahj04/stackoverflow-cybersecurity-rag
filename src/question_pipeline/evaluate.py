"""Evaluate the question pipeline end to end (search -> context) on the 500 test questions.

For each question: search the top 10 threads, build the context, then check
    thread rank       where the correct thread (gold or duplicate) is
    gold answer in context   a gold answer is among the answers kept for the correct thread
    accepted in context      the accepted answer is among them (when the thread has one)
and measure how big the context is (answers and approximate tokens).

Usage: python evaluate.py [limit]
"""
import json
import sys
import time
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent))
import context
import search

SRC_DIR = context.SRC_DIR
TEST_SET = SRC_DIR / "test_set_human.jsonl"
RESULTS_FILE = Path(__file__).resolve().parent / "results.json"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None

store = context.ctx.ThreadStore()
encoding = tiktoken.get_encoding("cl100k_base")
KS = (1, 3, 5, 10)


def evaluate_case(case):
    thread_ids = [t for t, _ in search.search_threads(case["query"], top_k=context.MAX_THREADS)]
    blocks = context.build_context(thread_ids, store)
    correct = {case["gold_thread_id"], *case["also_correct_threads"]}
    thread_rank = next((b["rank"] for b in blocks if b["thread_id"] in correct), None)
    kept = [a["answer_id"] for b in blocks if b["thread_id"] in correct for a in b["answers"]]
    accepted = case["accepted_answer_id"]
    return {
        "thread_rank": thread_rank,
        "gold_in_context": any(a in case["gold_answer_ids"] for a in kept),
        "accepted_in_context": (accepted in kept) if accepted else None,
        "n_answers": sum(len(b["answers"]) for b in blocks),
        "tokens": len(encoding.encode(context.format_context(blocks))),
    }


def share(values):
    values = list(values)
    return sum(values) / len(values)


def report(cases, results):
    n = len(results)
    print(f"\nQuestion pipeline, {n} questions (top {context.MAX_THREADS} threads, "
          f"{context.MAX_ANSWERS_PER_THREAD} answers per thread)")
    for k in KS:
        print(f"  correct thread in top {k:>2}: {share(r['thread_rank'] is not None and r['thread_rank'] <= k for r in results):.2f}")
    print(f"  MRR: {share(1 / r['thread_rank'] if r['thread_rank'] else 0 for r in results):.3f}")
    found = [r for r in results if r["thread_rank"]]
    print(f"\n  when the correct thread is in the context ({len(found)} questions):")
    print(f"    a gold answer is kept:      {share(r['gold_in_context'] for r in found):.2f}")
    with_acc = [r for r in found if r["accepted_in_context"] is not None]
    print(f"    the accepted answer is kept: {share(r['accepted_in_context'] for r in with_acc):.2f}  (n={len(with_acc)})")
    print(f"\n  end to end, gold answer reaches the LLM: {share(r['thread_rank'] is not None and r['gold_in_context'] for r in results):.2f}")
    toks = sorted(r["tokens"] for r in results)
    print(f"  context size: {share(r['n_answers'] for r in results):.0f} answers, mean {share(toks):.0f} tokens "
          f"(median {toks[n // 2]}, p90 {toks[int(n * .9)]}, max {toks[-1]})")
    print("\n  R@10 by persona:")
    for p in ["expert", "developer", "search", "learner", "non_technical"]:
        sub = [r for c, r in zip(cases, results) if c["style"] == p]
        print(f"    {p:<14}{share(r['thread_rank'] is not None for r in sub):.2f}")


def main():
    cases = [json.loads(line) for line in open(TEST_SET, encoding="utf-8")]
    cases = cases[:LIMIT] if LIMIT else cases
    results, start = [], time.time()
    for i, case in enumerate(cases, 1):
        results.append(evaluate_case(case))
        if i % 100 == 0:
            print(f"{i}/{len(cases)}  {time.time() - start:.0f}s", flush=True)
    report(cases, results)
    if not LIMIT:
        RESULTS_FILE.write_text(json.dumps(
            [{"query": c["query"], "style": c["style"], "bucket": c["bucket"], **r} for c, r in zip(cases, results)],
            indent=2))
        print(f"\nWrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()
