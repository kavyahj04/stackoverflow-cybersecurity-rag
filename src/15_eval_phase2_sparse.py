"""Phase 2: evaluate sparse search (BM25) alone, and compare it with dense.

Same metrics as Phase 1 (judge/summarize are reused from 14_eval_phase1_dense).
On top, per query, it checks against the Phase 1 dense results:
    sparse-only   BM25 found the right thread in its 50, dense did not
    dense-only    the reverse
    both / neither
"both + dense-only + sparse-only" is the recall of the combined pool
(dense 50 + sparse 50) that reaches the cross-encoder: the Phase 3 number.

Usage: python 15_eval_phase2_sparse.py [test_file] [dense_results] [results_file] [answers|both]
    answers = BM25 over answers only; both = answers + questions (RRF)
"""
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
TEST_SET = SRC_DIR / (sys.argv[1] if len(sys.argv) > 1 else "test_set_human.jsonl")
DENSE_RESULTS = SRC_DIR / (sys.argv[2] if len(sys.argv) > 2 else "phase1_dense_human_results.json")
RESULTS_FILE = SRC_DIR / (sys.argv[3] if len(sys.argv) > 3 else "phase2_sparse_human_results.json")
USE_QUESTIONS = (sys.argv[4] if len(sys.argv) > 4 else "both") == "both"

TOP_K = 50

phase1 = importlib.import_module("14_eval_phase1_dense")
hybrid = phase1.hybrid


def load_cases():
    with open(TEST_SET, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def compare(cases, sparse_results, dense_results):
    """Who found the right thread within their 50: both, one, or neither."""
    rows = []
    for c, s, d in zip(cases, sparse_results, dense_results):
        found_s, found_d = s["thread_rank"] is not None, d["thread_rank"] is not None
        rows.append({
            "style": c["style"],
            "outcome": ("both" if found_s and found_d else "sparse_only" if found_s
                        else "dense_only" if found_d else "neither"),
            "sparse_better": bool(found_s and (not found_d or s["thread_rank"] < d["thread_rank"])),
        })
    return rows


def print_compare(rows):
    n = len(rows)
    counts = Counter(r["outcome"] for r in rows)
    print("\nSparse vs dense (right thread anywhere in their 50):")
    for k in ["both", "dense_only", "sparse_only", "neither"]:
        print(f"  {k:<12} {counts[k]:>4}  ({counts[k] / n:.0%})")
    pool = n - counts["neither"]
    print(f"  combined pool recall (dense 50 + sparse 50): {pool}/{n} = {pool / n:.2f}")
    print(f"  sparse ranked it higher than dense: {sum(r['sparse_better'] for r in rows)}")
    print("\n  sparse-only finds by persona:",
          dict(Counter(r["style"] for r in rows if r["outcome"] == "sparse_only")))


def main():
    cases = load_cases()
    dense = json.loads(DENSE_RESULTS.read_text())["per_query"]
    assert [d["query"] for d in dense] == [c["query"] for c in cases], "dense results don't match test set"

    results = []
    for i, case in enumerate(cases, 1):
        hits = hybrid.sparse_search(case["query"], TOP_K, use_questions=USE_QUESTIONS)
        results.append(phase1.judge(case, hits))
        if i % 100 == 0:
            print(f"{i}/{len(cases)}", flush=True)

    summary = phase1.summarize(results)
    phase1.print_main(summary, f"Sparse search (BM25, {'answers + questions' if USE_QUESTIONS else 'answers only'})")
    by_bucket, by_style = {}, {}
    for c, r in zip(cases, results):
        by_bucket.setdefault(c["bucket"], []).append(r)
        by_style.setdefault(c["style"], []).append(r)
    phase1.print_split("bucket", by_bucket)
    phase1.print_split("style", by_style)
    phase1.print_depth(results)

    rows = compare(cases, results, dense)
    print_compare(rows)

    RESULTS_FILE.write_text(json.dumps({
        "summary": summary,
        "by_bucket": {b: phase1.summarize(rs) for b, rs in by_bucket.items()},
        "by_style": {s: phase1.summarize(rs) for s, rs in by_style.items()},
        "vs_dense": dict(Counter(r["outcome"] for r in rows)),
        "per_query": [{"query": c["query"], "bucket": c["bucket"], "style": c["style"], **r,
                       "vs_dense": cmp["outcome"]}
                      for c, r, cmp in zip(cases, results, rows)],
    }, indent=2))
    print(f"\nWrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()
