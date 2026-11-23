"""Evaluate search -> context on the human-style test set.

For each test query: search the top threads, build the context, then record
    thread rank          where the correct thread (gold or a duplicate) is, or None
    gold_in_context      a gold answer is among the answers kept for that thread
    accepted_in_context  the accepted answer is among them (when the thread has one)
plus the context size. Test cases whose thread is no longer in the database
(removed by the close-reason filter) are skipped and counted.

Every run writes reports/retrieval_eval_<time>.json with the settings, code
version and test-set hash, so any score can be traced back and reproduced.

Usage (from rag-dev):  python -m rag.eval.retrieval_eval [--limit N]
"""
import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from datetime import datetime

import tiktoken

from rag.config import REPORTS_DIR, ROOT_DIR, TEST_SET_PATH, get_settings
from rag.context import build_context, format_context
from rag.db import connect
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.store import ThreadStore

KS = (1, 3, 5, 10)
PERSONAS = ("expert", "developer", "search", "learner", "non_technical")
encoding = tiktoken.get_encoding("cl100k_base")  # approximate token count of the context


def load_cases(path):
    """Test cases with ids converted from text to numbers, matching the database."""
    cases = []
    for line in open(path, encoding="utf-8"):
        c = json.loads(line)
        c["gold_thread_id"] = int(c["gold_thread_id"])
        c["also_correct_threads"] = [int(t) for t in c["also_correct_threads"]]
        c["gold_answer_ids"] = [int(a) for a in c["gold_answer_ids"]]
        c["accepted_answer_id"] = int(c["accepted_answer_id"]) if c["accepted_answer_id"] else None
        cases.append(c)
    return cases


def available_cases(cases, store):
    """Keep test cases whose correct thread (or a duplicate of it) is still in the database."""
    ids = {t for c in cases for t in [c["gold_thread_id"], *c["also_correct_threads"]]}
    present = store.existing_ids(ids)
    return [c for c in cases if {c["gold_thread_id"], *c["also_correct_threads"]} & present]


def evaluate_case(case, retriever, store):
    hits = retriever.search(case["query"])
    question_ids = [qid for qid, _ in hits]
    blocks = build_context(question_ids, store)
    correct = {case["gold_thread_id"], *case["also_correct_threads"]}
    thread_rank = next((b["rank"] for b in blocks if b["question_id"] in correct), None)
    kept = [a["answer_id"] for b in blocks if b["question_id"] in correct for a in b["answers"]]
    accepted = case["accepted_answer_id"]
    return {
        "query": case["query"],
        "style": case["style"],
        "bucket": case["bucket"],
        "gold_thread_id": case["gold_thread_id"],
        "retrieved": question_ids,
        "thread_rank": thread_rank,
        "gold_in_context": any(a in case["gold_answer_ids"] for a in kept),
        "accepted_in_context": (accepted in kept) if accepted else None,
        "n_answers": sum(len(b["answers"]) for b in blocks),
        "tokens": len(encoding.encode(format_context(blocks))),
    }


def share(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarise(results):
    found = [r for r in results if r["thread_rank"]]
    with_accepted = [r for r in found if r["accepted_in_context"] is not None]
    tokens = sorted(r["tokens"] for r in results)
    n = len(results)
    return {
        "n": n,
        **{f"recall@{k}": share(r["thread_rank"] is not None and r["thread_rank"] <= k for r in results) for k in KS},
        "mrr": share(1 / r["thread_rank"] if r["thread_rank"] else 0 for r in results),
        "gold_answer_kept_when_found": share(r["gold_in_context"] for r in found),
        "accepted_kept_when_found": share(r["accepted_in_context"] for r in with_accepted),
        "context_answers_mean": share(r["n_answers"] for r in results),
        "context_tokens": {"mean": share(tokens), "median": tokens[n // 2], "p90": tokens[int(n * .9)], "max": tokens[-1]},
        "recall@10_by_persona": {p: share(r["thread_rank"] is not None for r in results if r["style"] == p)
                                 for p in PERSONAS},
    }


def print_summary(m, skipped):
    print(f"\nRetrieval eval: {m['n']} queries ({skipped} skipped: thread not in database)")
    for k in KS:
        print(f"  correct thread in top {k:>2}: {m[f'recall@{k}']:.3f}")
    print(f"  MRR: {m['mrr']:.3f}")
    print(f"  gold answer kept when thread found:     {m['gold_answer_kept_when_found']:.2f}")
    print(f"  accepted answer kept when thread found: {m['accepted_kept_when_found']:.2f}")
    t = m["context_tokens"]
    print(f"  context: {m['context_answers_mean']:.0f} answers, mean {t['mean']:.0f} tokens "
          f"(median {t['median']}, p90 {t['p90']}, max {t['max']})")
    print("  recall@10 by persona:")
    for p, v in m["recall@10_by_persona"].items():
        print(f"    {p:<14}{v:.2f}")


def git_version():
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT_DIR, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "--short", "HEAD"), "uncommitted_changes": bool(run("status", "--porcelain"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="only evaluate the first N queries (quick check, no report)")
    args = parser.parse_args()

    conn = connect()
    store = ThreadStore(conn)
    retriever = DenseRetriever(conn)

    all_cases = load_cases(TEST_SET_PATH)
    cases = available_cases(all_cases, store)
    skipped = len(all_cases) - len(cases)
    if args.limit:
        cases = cases[:args.limit]

    results, start = [], time.time()
    for i, case in enumerate(cases, 1):
        results.append(evaluate_case(case, retriever, store))
        if i % 100 == 0:
            print(f"{i}/{len(cases)}  {time.time() - start:.0f}s", flush=True)

    metrics = summarise(results)
    print_summary(metrics, skipped)

    if not args.limit:
        REPORTS_DIR.mkdir(exist_ok=True)
        settings = asdict(get_settings())
        settings.pop("database_url")  # never write the connection string (it holds the password)
        report = {
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "code": git_version(),
            "settings": settings,
            "test_set": {"path": str(TEST_SET_PATH.relative_to(ROOT_DIR)),
                         "sha256": hashlib.sha256(TEST_SET_PATH.read_bytes()).hexdigest()[:16],
                         "cases_total": len(all_cases), "cases_skipped": skipped},
            "duration_s": round(time.time() - start),
            "metrics": metrics,
            "cases": results,
        }
        path = REPORTS_DIR / f"retrieval_eval_{datetime.now():%Y%m%d-%H%M%S}.json"
        path.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
