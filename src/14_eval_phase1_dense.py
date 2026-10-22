"""Phase 1: evaluate dense search alone on test_set.jsonl.

For each test query, dense_search returns 50 answers. We record:
    thread rank      rank of the first thread that is gold_thread_id or in
                     also_correct_threads (threads in first-seen order)
    good-answer pos  position of the first answer in gold_answer_ids
    accepted pos     position of accepted_answer_id (when the thread has one)

Metrics: recall@k and MRR on thread rank; good-answer@k and accepted@k on
answer positions. Results are also split by bucket and style, and the rank
distribution shows how many dense results are actually needed.
"""
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
# Usage: python 14_eval_phase1_dense.py [test_file] [results_file]
TEST_SET = SRC_DIR / (sys.argv[1] if len(sys.argv) > 1 else "test_set.jsonl")
RESULTS_FILE = SRC_DIR / (sys.argv[2] if len(sys.argv) > 2 else "phase1_dense_results.json")

TOP_K = 50
KS = (3, 5, 10, 20, 30, 50)

hybrid = importlib.import_module("08_hybrid_search")
ctx = importlib.import_module("12_build_context")
store = ctx.ThreadStore()


def load_cases():
    with open(TEST_SET, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def judge(case, hits):
    """Positions (1-based) of the first correct thread / good answer / accepted answer."""
    answer_ids = [hit["answer_id"] for hit in hits]
    answer_to_thread = store.thread_ids_for(answer_ids)
    correct_threads = {case["gold_thread_id"], *case["also_correct_threads"]}

    thread_order = []
    for aid in answer_ids:
        thread_id = answer_to_thread[aid]
        if thread_id not in thread_order:
            thread_order.append(thread_id)

    def first(items, wanted):
        return next((i for i, item in enumerate(items, 1) if item in wanted), None)

    accepted = case["accepted_answer_id"]
    return {
        "thread_rank": first(thread_order, correct_threads),
        "good_answer_pos": first(answer_ids, set(case["gold_answer_ids"])),
        "accepted_pos": first(answer_ids, {accepted}) if accepted else None,
        "has_accepted": bool(accepted),
        "top_threads": thread_order[:3],
    }


def at_k(values, k):
    return sum(v is not None and v <= k for v in values) / max(len(values), 1)


def summarize(results):
    ranks = [r["thread_rank"] for r in results]
    good = [r["good_answer_pos"] for r in results]
    accepted = [r["accepted_pos"] for r in results if r["has_accepted"]]
    summary = {"queries": len(results)}
    summary.update({f"recall@{k}": at_k(ranks, k) for k in KS})
    summary["mrr"] = sum(1 / r for r in ranks if r) / len(ranks)
    summary.update({f"good_answer@{k}": at_k(good, k) for k in KS})
    summary.update({f"accepted@{k}": at_k(accepted, k) for k in KS})
    summary["accepted_queries"] = len(accepted)
    return summary


def print_main(summary, label="Dense search"):
    print(f"\n{label}, {summary['queries']} queries (thread = right question found)")
    print(f"{'k':>4} {'recall@k':>10} {'good_answer@k':>15} {'accepted@k':>12}")
    for k in KS:
        print(f"{k:>4} {summary[f'recall@{k}']:>10.2f} {summary[f'good_answer@{k}']:>15.2f} "
              f"{summary[f'accepted@{k}']:>12.2f}")
    print(f"MRR = {summary['mrr']:.3f}   (accepted@k over {summary['accepted_queries']} queries)")


def print_split(title, groups):
    print(f"\nBy {title}:")
    print(f"{'':<14} {'n':>3} {'R@5':>6} {'R@10':>6} {'R@20':>6} {'R@50':>6} {'MRR':>6}")
    for name, rows in groups.items():
        s = summarize(rows)
        print(f"{name:<14} {s['queries']:>3} {s['recall@5']:>6.2f} {s['recall@10']:>6.2f} "
              f"{s['recall@20']:>6.2f} {s['recall@50']:>6.2f} {s['mrr']:>6.2f}")


def print_depth(results):
    ranks = sorted(r["thread_rank"] for r in results if r["thread_rank"])
    bins = Counter("1" if r == 1 else "2-5" if r <= 5 else "6-10" if r <= 10
                   else "11-20" if r <= 20 else "21-30" if r <= 30 else "31+" for r in ranks)
    print(f"\nWhere the right thread was found (thread rank), {len(ranks)} of {len(results)} found:")
    for b in ["1", "2-5", "6-10", "11-20", "21-30", "31+"]:
        print(f"  rank {b:<6} {bins.get(b, 0):>3}")
    print(f"  not found {len(results) - len(ranks):>3}")
    if ranks:
        print(f"  deepest hit: thread rank {ranks[-1]}")


def print_misses(cases, results, limit=10):
    titles = dict(store.conn.execute("SELECT question_id, title FROM questions"))
    misses = [(c, r) for c, r in zip(cases, results)
              if r["thread_rank"] is None or r["thread_rank"] > 10]
    print(f"\nMisses (right thread not in top 10): {len(misses)}; first {min(limit, len(misses))}:")
    for c, r in misses[:limit]:
        print(f"\n  query : {c['query']}")
        print(f"  gold  : {c['thread_title']}  (found at rank {r['thread_rank']})")
        for t in r["top_threads"]:
            print(f"  got   : {titles.get(t, t)}")


def main():
    cases = load_cases()
    results = []
    for i, case in enumerate(cases, 1):
        hits = hybrid.dense_search(case["query"], TOP_K)
        results.append(judge(case, hits))
        if i % 100 == 0:
            print(f"{i}/{len(cases)}", flush=True)

    summary = summarize(results)
    print_main(summary)
    by_bucket, by_style = {}, {}
    for c, r in zip(cases, results):
        by_bucket.setdefault(c["bucket"], []).append(r)
        by_style.setdefault(c["style"], []).append(r)
    print_split("bucket", by_bucket)
    print_split("style", by_style)
    print_depth(results)
    print_misses(cases, results)

    RESULTS_FILE.write_text(json.dumps({
        "summary": summary,
        "by_bucket": {b: summarize(rs) for b, rs in by_bucket.items()},
        "by_style": {s: summarize(rs) for s, rs in by_style.items()},
        "per_query": [{"query": c["query"], "bucket": c["bucket"], "style": c["style"], **r}
                      for c, r in zip(cases, results)],
    }, indent=2))
    print(f"\nWrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()
