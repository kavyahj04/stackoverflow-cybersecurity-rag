"""Retrieval variants without the cross-encoder, all at thread level, on every test question.

Each variant turns a search into a ranked list of threads; RRF merges lists.
Variants:
    dense_answers_50   dense over the answers collection (baseline, Phase 1)
    dense_answers_30   same list cut to its first 30 answers' threads
    dense_questions    dense over the questions collection (title + body)
    dense_a+q          RRF of the two dense lists
    dense_a+bm25       RRF of dense answers and BM25 (answers + questions)
    dense_a+q+bm25     RRF of all three

Usage: python 17_eval_fusion.py [limit]
"""
import importlib
import json
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
TEST_SET = SRC_DIR / "test_set_human.jsonl"
RESULTS_FILE = SRC_DIR / "fusion_results.json"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None
TOP_K = 50
RRF_K = 60
KS = (3, 5, 10, 20, 50)

hybrid = importlib.import_module("08_hybrid_search")
ctx = importlib.import_module("12_build_context")
store = ctx.ThreadStore()
questions_collection = hybrid.chroma_client.get_collection("questions")


def threads_from_answers(answer_ids):
    mapping = store.thread_ids_for(answer_ids)
    ordered = []
    for aid in answer_ids:
        if mapping[aid] not in ordered:
            ordered.append(mapping[aid])
    return ordered


def rrf(*ranked_lists):
    scores = {}
    for ranked in ranked_lists:
        for rank, thread_id in enumerate(ranked, 1):
            scores[thread_id] = scores.get(thread_id, 0) + 1 / (RRF_K + rank)
    return sorted(scores, key=scores.get, reverse=True)


def searches(query):
    embedding = hybrid.embed_query(query)
    answers = questions = None
    a = hybrid.answers_collection.query(query_embeddings=[embedding], n_results=TOP_K)["ids"][0]
    q = questions_collection.query(query_embeddings=[embedding], n_results=TOP_K)["ids"][0]
    bm = [h["answer_id"] for h in hybrid.sparse_search(query, TOP_K)]
    return a, q, bm


def variants(a_ids, q_ids, bm_ids):
    dense_a = threads_from_answers(a_ids)
    dense_a30 = threads_from_answers(a_ids[:30])
    bm = threads_from_answers(bm_ids)
    return {
        "dense_answers_50": dense_a,
        "dense_answers_30": dense_a30,
        "dense_questions": q_ids,
        "dense_a+q": rrf(dense_a, q_ids),
        "dense_a+bm25": rrf(dense_a, bm),
        "dense_a+q+bm25": rrf(dense_a, q_ids, bm),
    }


def rank_of(threads, case):
    correct = {case["gold_thread_id"], *case["also_correct_threads"]}
    return next((i for i, t in enumerate(threads, 1) if t in correct), None)


def summarize(ranks):
    n = len(ranks)
    out = {f"R@{k}": sum(r is not None and r <= k for r in ranks) / n for k in KS}
    out["MRR"] = sum(1 / r for r in ranks if r) / n
    return out


def main():
    cases = [json.loads(line) for line in open(TEST_SET, encoding="utf-8")]
    cases = cases[:LIMIT] if LIMIT else cases
    ranks, start = {}, time.time()
    for i, case in enumerate(cases, 1):
        for name, threads in variants(*searches(case["query"])).items():
            ranks.setdefault(name, []).append(rank_of(threads, case))
        if i % 50 == 0:
            print(f"{i}/{len(cases)}  {time.time() - start:.0f}s", flush=True)

    names = list(ranks)
    print(f"\n{len(cases)} questions (thread-level recall)")
    print(f"{'variant':<20}" + "".join(f"{k:>8}" for k in [f"R@{k}" for k in KS] + ["MRR"]))
    summaries = {n: summarize(ranks[n]) for n in names}
    for n in names:
        print(f"{n:<20}" + "".join(f"{summaries[n][k]:>8.2f}" for k in [f"R@{k}" for k in KS] + ["MRR"]))

    styles = ["expert", "developer", "search", "learner", "non_technical"]
    print(f"\nR@10 by persona{'':<5}" + "".join(f"{s[:9]:>11}" for s in styles))
    for n in names:
        row = []
        for s in styles:
            idx = [j for j, c in enumerate(cases) if c["style"] == s]
            row.append(sum(ranks[n][j] is not None and ranks[n][j] <= 10 for j in idx) / len(idx))
        print(f"{n:<20}" + "".join(f"{v:>11.2f}" for v in row))

    RESULTS_FILE.write_text(json.dumps({"summaries": summaries, "ranks": ranks}, indent=2))
    print(f"\nWrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()
