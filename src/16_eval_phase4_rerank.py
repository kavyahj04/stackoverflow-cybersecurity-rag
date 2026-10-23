"""Phase 4: evaluate hybrid_search (dense top 5 kept + cross-encoder reranked fill).

Runs on a fixed sample (4 threads from each vote bucket = 20 threads, all 5
personas = 100 questions) because the cross-encoder is slow. Compared against
dense and sparse on the same questions using the Phase 1/2 result files.

Per question we record where the right thread lands in the final 50 answers and
whether the reranker rescued it (not in dense top 10 -> in hybrid top 10),
lost it (the reverse), or moved it. For rescues we also record where it came
from: dense rank 11-20, 21-50, or only from BM25.

Usage: python 16_eval_phase4_rerank.py [limit|all]   (limit = N questions for timing; all = every test question)
Resumes from the results file if it already has some questions done.
"""
import importlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
TEST_SET = SRC_DIR / "test_set_human.jsonl"
DENSE_RESULTS = SRC_DIR / "phase1_dense_human_results.json"
SPARSE_RESULTS = SRC_DIR / "phase2_sparse_stem_both.json"
RESULTS_FILE = SRC_DIR / "phase4_rerank_results.json"
ARG = sys.argv[1] if len(sys.argv) > 1 else None
ALL = ARG == "all"
LIMIT = int(ARG) if ARG and not ALL else None

THREADS_PER_BUCKET = 4
SEED = 42
TOP_K = 50

phase1 = importlib.import_module("14_eval_phase1_dense")
hybrid = phase1.hybrid


def load_sample():
    cases = [json.loads(line) for line in open(TEST_SET, encoding="utf-8")]
    dense = json.loads(DENSE_RESULTS.read_text())["per_query"]
    sparse = json.loads(SPARSE_RESULTS.read_text())["per_query"]
    if ALL:
        return list(zip(cases, dense, sparse))
    rng = random.Random(SEED)
    chosen = set()
    for bucket in dict.fromkeys(c["bucket"] for c in cases):
        threads = sorted({c["gold_thread_id"] for c in cases if c["bucket"] == bucket})
        chosen |= set(rng.sample(threads, THREADS_PER_BUCKET))
    rows = [(c, d, s) for c, d, s in zip(cases, dense, sparse) if c["gold_thread_id"] in chosen]
    return rows[:LIMIT] if LIMIT else rows


def outcome(dense_rank, hybrid_rank):
    d10 = dense_rank is not None and dense_rank <= 10
    h10 = hybrid_rank is not None and hybrid_rank <= 10
    if h10 and not d10:
        return "rescued"
    if d10 and not h10:
        return "lost"
    if dense_rank is None and hybrid_rank is None:
        return "never_found"
    if dense_rank == hybrid_rank:
        return "same"
    if hybrid_rank is not None and (dense_rank is None or hybrid_rank < dense_rank):
        return "moved_up"
    return "moved_down"


def rescue_source(dense_rank, sparse_rank):
    if dense_rank is None:
        return "BM25 only" if sparse_rank is not None else "neither"
    return "dense 11-20" if dense_rank <= 20 else "dense 21-50"


def print_report(rows):
    n = len(rows)
    hy = [r["hybrid"] for r in rows]
    de = [r["dense"] for r in rows]
    print(f"\nPhase 4, {n} questions ({n // 5} threads x 5 personas)")
    print(f"{'':<14}" + "".join(f"{c:>8}" for c in ["R@3", "R@5", "R@10", "R@20", "R@50", "MRR", "GA@10"]))
    for name, res in [("dense", de), ("hybrid", hy)]:
        s = phase1.summarize(res)
        print(f"{name:<14}" + "".join(f"{s[k]:>8.2f}" for k in
              ["recall@3", "recall@5", "recall@10", "recall@20", "recall@50", "mrr", "good_answer@10"]))
    print("\nCompared with dense, per question:", dict(Counter(r["outcome"] for r in rows)))
    rescued = [r for r in rows if r["outcome"] == "rescued"]
    print("rescued into top 10 came from:", dict(Counter(r["rescue_source"] for r in rescued)))
    print("\nBy persona (R@10 dense -> hybrid):")
    for p in ["expert", "developer", "search", "learner", "non_technical"]:
        sub = [r for r in rows if r["style"] == p]
        if not sub:
            continue
        a = sum(r["dense"]["thread_rank"] is not None and r["dense"]["thread_rank"] <= 10 for r in sub) / len(sub)
        b = sum(r["hybrid"]["thread_rank"] is not None and r["hybrid"]["thread_rank"] <= 10 for r in sub) / len(sub)
        print(f"  {p:<14} {a:.2f} -> {b:.2f}  (n={len(sub)})")
    t = [r["seconds"] for r in rows]
    print(f"\nTime per question: mean {sum(t) / len(t):.0f}s, max {max(t):.0f}s")


def main():
    sample = load_sample()
    done = {}
    if RESULTS_FILE.exists() and not LIMIT:
        done = {r["query"]: r for r in json.loads(RESULTS_FILE.read_text()).get("per_query", [])}
    rows = []
    for i, (case, dense, sparse) in enumerate(sample, 1):
        if case["query"] in done:
            rows.append(done[case["query"]])
            continue
        start = time.time()
        final = hybrid.hybrid_search(case["query"], top_k=TOP_K)["final"]
        result = phase1.judge(case, final)
        rows.append({
            "query": case["query"], "style": case["style"], "bucket": case["bucket"],
            "dense": {k: dense[k] for k in ["thread_rank", "good_answer_pos", "accepted_pos", "has_accepted"]},
            "sparse_thread_rank": sparse["thread_rank"],
            "hybrid": {k: result[k] for k in ["thread_rank", "good_answer_pos", "accepted_pos", "has_accepted"]},
            "outcome": outcome(dense["thread_rank"], result["thread_rank"]),
            "rescue_source": rescue_source(dense["thread_rank"], sparse["thread_rank"]),
            "seconds": round(time.time() - start, 1),
        })
        print(f"{i}/{len(sample)}  {rows[-1]['seconds']:.0f}s  {rows[-1]['outcome']}", flush=True)
        if not LIMIT and i % 10 == 0:
            RESULTS_FILE.write_text(json.dumps({"partial": True, "per_query": rows}, indent=2))

    print_report(rows)
    if not LIMIT:
        RESULTS_FILE.write_text(json.dumps({"partial": False, "per_query": rows}, indent=2))
        print(f"\nWrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()
