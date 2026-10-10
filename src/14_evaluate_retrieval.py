"""Retrieval evaluation on the duplicate-question test set (eval_set.jsonl).

Every method's output is turned into a ranked list of threads, because a hit
means finding the right thread, whichever of its answers came back.

Metrics (per method):
    recall@k      share of queries with a gold thread in the top k threads
    MRR           mean of 1/rank of the first gold thread (0 if not found)
    accepted@k    share of queries (with an accepted gold answer) where that
                  accepted answer is among the answers of the top k threads

Phase A: dense + sparse on all queries (cheap).
Phase B: dense, sparse, hybrid and build_context on a fixed sample (the
         cross-encoder makes these slow), so all four are compared on the
         same queries.
"""
import importlib
import json
import random
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
EVAL_SET = SRC_DIR / "eval_set.jsonl"
RESULTS_FILE = SRC_DIR / "eval_results.json"

KS = (3, 5, 10)
TOP_K = 50
SAMPLE_SIZE = 150
SEED = 42

hybrid = importlib.import_module("08_hybrid_search")
ctx = importlib.import_module("12_build_context")
store = ctx.ThreadStore()


def load_cases():
    with open(EVAL_SET, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def threads_from_hits(hits):
    """Answer hits -> [(thread_id, [answer_ids...]), ...] in first-seen order."""
    answer_to_thread = store.thread_ids_for(hit["answer_id"] for hit in hits)
    ranked = {}
    for hit in hits:
        ranked.setdefault(answer_to_thread[hit["answer_id"]], []).append(hit["answer_id"])
    return list(ranked.items())


def threads_from_blocks(blocks):
    return [(b["thread_id"], [a["answer_id"] for a in b["answers"]]) for b in blocks]


def judge(ranked_threads, case):
    """Where did this method put the gold thread / accepted answer?"""
    gold = set(case["gold_thread_ids"])
    accepted = set(case["gold_accepted_ids"])
    thread_rank = next((i for i, (t, _) in enumerate(ranked_threads, 1) if t in gold), None)
    accepted_rank = next(
        (i for i, (_, answers) in enumerate(ranked_threads, 1) if accepted & set(answers)), None
    )
    return {"thread_rank": thread_rank, "accepted_rank": accepted_rank}


def summarize(case_results, cases):
    n = len(case_results)
    with_accepted = [r for r, c in zip(case_results, cases) if c["gold_accepted_ids"]]
    summary = {"queries": n}
    for k in KS:
        summary[f"recall@{k}"] = sum(r["thread_rank"] is not None and r["thread_rank"] <= k
                                     for r in case_results) / n
    summary["mrr"] = sum(1 / r["thread_rank"] for r in case_results if r["thread_rank"]) / n
    for k in KS:
        summary[f"accepted@{k}"] = sum(r["accepted_rank"] is not None and r["accepted_rank"] <= k
                                       for r in with_accepted) / max(len(with_accepted), 1)
    return summary


def with_retry(fn, *args, attempts=3):
    for attempt in range(attempts):
        try:
            return fn(*args)
        except Exception as exc:  # network hiccups on the embedding API
            if attempt == attempts - 1:
                raise
            print(f"  retrying after error: {exc}")
            time.sleep(5)


def run_phase_a(cases):
    per_method = {"dense": [], "sparse": []}
    for i, case in enumerate(cases, 1):
        dense = with_retry(hybrid.dense_search, case["query"], TOP_K)
        sparse = with_retry(hybrid.sparse_search, case["query"], TOP_K)
        per_method["dense"].append(judge(threads_from_hits(dense), case))
        per_method["sparse"].append(judge(threads_from_hits(sparse), case))
        if i % 100 == 0:
            print(f"phase A: {i}/{len(cases)}", flush=True)
    return per_method


def run_phase_b(cases):
    per_method = {"dense": [], "sparse": [], "hybrid": [], "build_context": []}
    start = time.time()
    for i, case in enumerate(cases, 1):
        q = case["query"]
        results = with_retry(hybrid.hybrid_search, q, TOP_K)
        blocks = ctx.build_context(q, results["final"], store, hybrid.cross_encoder)
        per_method["dense"].append(judge(threads_from_hits(results["dense"]), case))
        per_method["sparse"].append(judge(threads_from_hits(results["sparse"]), case))
        per_method["hybrid"].append(judge(threads_from_hits(results["final"]), case))
        per_method["build_context"].append(judge(threads_from_blocks(blocks), case))
        per_query = (time.time() - start) / i
        print(f"phase B: {i}/{len(cases)}  ({per_query:.0f}s/query, "
              f"~{per_query * (len(cases) - i) / 60:.0f} min left)", flush=True)
    return per_method


def print_table(title, summaries):
    cols = ["recall@3", "recall@5", "recall@10", "mrr", "accepted@3", "accepted@5", "accepted@10"]
    print(f"\n{title}")
    print(f"{'method':<15}" + "".join(f"{c:>12}" for c in cols))
    for method, s in summaries.items():
        print(f"{method:<15}" + "".join(f"{s[c]:>12.3f}" for c in cols))


def main():
    cases = load_cases()
    sample = random.Random(SEED).sample(cases, SAMPLE_SIZE)
    output = {"settings": {"top_k": TOP_K, "sample_size": SAMPLE_SIZE, "seed": SEED,
                           "cross_encoder": hybrid.cross_encoder.MODEL_NAME,
                           "max_threads": ctx.MAX_THREADS,
                           "max_answers_per_thread": ctx.MAX_ANSWERS_PER_THREAD}}

    phase_a = run_phase_a(cases)
    output["phase_a"] = {m: summarize(r, cases) for m, r in phase_a.items()}
    print_table(f"Phase A: all {len(cases)} queries", output["phase_a"])
    RESULTS_FILE.write_text(json.dumps(output, indent=2))

    phase_b = run_phase_b(sample)
    output["phase_b"] = {m: summarize(r, sample) for m, r in phase_b.items()}
    output["phase_b_per_query"] = [
        {"query": c["query"], "gold_thread_ids": c["gold_thread_ids"],
         **{m: phase_b[m][i] for m in phase_b}}
        for i, c in enumerate(sample)
    ]
    print_table(f"Phase B: sample of {len(sample)} queries", output["phase_b"])
    RESULTS_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()
