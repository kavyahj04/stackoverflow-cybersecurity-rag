"""Evaluate generated answers on a fixed sample of the human test set.

Every configuration answers the same questions from the same search results
(search runs once per question), so differences come from the prompt or the
context size only. Each answer is graded by the judge (rag/eval/judge.py):
faithfulness, citation accuracy, answer relevance, reference coverage.
Scores are reported with 95% bootstrap intervals, and each configuration is
compared with the first one question by question (paired), which separates
real differences from luck far better than comparing two averages.

Writes reports/answer_eval_<time>.json with settings, code version, sample,
every answer and every judge verdict, so any score can be checked by hand.

Usage (from rag-dev):
    python -m rag.eval.answer_eval [--n 100] [--configs v1_10x3 v2_10x3 ...] [--workers 6]
"""
import argparse
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime

import numpy as np

from rag.config import REPORTS_DIR, ROOT_DIR, TEST_SET_PATH, get_settings
from rag.db import connect
from rag.eval.judge import Judge
from rag.eval.retrieval_eval import available_cases, git_version, load_cases
from rag.generation.answer import RagPipeline
from rag.generation.prompt import EXAMPLE_NAMES
from rag.retrieval.bm25 import BM25
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.store import ThreadStore

CONFIGS = {  # name: prompt and context settings (the first one listed is the baseline)
    "v1_10x3": {"few_shot": False, "max_threads": 10, "max_answers": 3},
    "v2_10x3": {"few_shot": True, "max_threads": 10, "max_answers": 3},
    "v2_7x2": {"few_shot": True, "max_threads": 7, "max_answers": 2},
    "v2_5x3": {"few_shot": True, "max_threads": 5, "max_answers": 3},
}
METRICS = ("faithfulness", "citation_accuracy", "answer_relevance", "reference_coverage")


def sample_cases(cases, n, seed):
    """The same n questions every run: an equal number per persona, chosen with a fixed seed."""
    rng = random.Random(seed)
    styles = sorted({c["style"] for c in cases})
    chosen = []
    for style in styles:
        group = [c for c in cases if c["style"] == style]
        chosen += rng.sample(group, n // len(styles))
    return chosen


def bootstrap_ci(values, seed=0, rounds=2000):
    values = np.array([v for v in values if v is not None], dtype=float)
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (rounds, len(values))).mean(axis=1)
    return {"mean": round(float(values.mean()), 3), "ci95": [round(float(np.percentile(means, 2.5)), 3),
                                                            round(float(np.percentile(means, 97.5)), 3)],
            "n": int(len(values))}


def grade(judge, case, answer):
    text = answer.text
    results = {
        "faithfulness": judge.faithfulness(text, answer.context_sources),
        "citation_accuracy": judge.citation_accuracy(text, answer.context_sources),
        "answer_relevance": judge.answer_relevance(case["query"], text),
        "reference_coverage": judge.reference_coverage(case["reference_answer"], text),
    }
    return {
        "query": case["query"],
        "style": case["style"],
        "gold_thread_id": case["gold_thread_id"],
        "answer": text,
        "cited": answer.citations.cited,
        "invalid_citations": answer.citations.invalid,
        "uncited_sentences": answer.citations.uncited_sentences,
        "example_leak": any(name in text for name in EXAMPLE_NAMES),
        "threads": answer.retrieval.get("threads"),                         # what search returned
        "context_answer_ids": [s.answer_id for s in answer.context_sources],  # source [n] = item n-1
        "threads_in_context": answer.retrieval.get("threads_in_context"),
        "usage": answer.usage,
        "scores": {m: results[m]["score"] for m in METRICS},
        "judge": {m: {k: v for k, v in results[m].items() if k not in ("score",)} for m in METRICS},
    }


def summarise(rows, baseline_rows):
    summary = {m: bootstrap_ci([r["scores"][m] for r in rows]) for m in METRICS}
    if baseline_rows is not None:
        base = {r["query"]: r for r in baseline_rows}
        for m in METRICS:
            diffs = [r["scores"][m] - base[r["query"]]["scores"][m] for r in rows
                     if r["scores"][m] is not None and base[r["query"]]["scores"][m] is not None]
            summary[f"{m}_vs_baseline"] = bootstrap_ci(diffs)
    summary.update({
        "uncited_sentences_mean": round(float(np.mean([r["uncited_sentences"] for r in rows])), 2),
        "invalid_citations_total": sum(len(r["invalid_citations"]) for r in rows),
        "example_leaks": sum(r["example_leak"] for r in rows),
        "sources_cited_mean": round(float(np.mean([len(r["cited"]) for r in rows])), 1),
        "prompt_tokens_mean": round(float(np.mean([r["usage"]["prompt_tokens"] for r in rows]))),
        "completion_tokens_mean": round(float(np.mean([r["usage"]["completion_tokens"] for r in rows]))),
        "answer_seconds_mean": round(float(np.mean([r["usage"]["seconds"] for r in rows])), 1),
        "cut_off": sum(r["usage"]["cut_off"] for r in rows),
    })
    return summary


def print_table(summaries):
    print(f"\n{'config':<10}" + "".join(f"{m:>22}" for m in METRICS) + f"{'tokens in':>11}{'uncited':>9}{'leaks':>7}")
    for name, s in summaries.items():
        cells = ""
        for m in METRICS:
            ci = s[m]
            cells += f"{ci['mean']:>8.3f} [{ci['ci95'][0]:.2f},{ci['ci95'][1]:.2f}]" if ci else f"{'-':>22}"
        print(f"{name:<10}{cells}{s['prompt_tokens_mean']:>11}{s['uncited_sentences_mean']:>9}{s['example_leaks']:>7}")
    names = list(summaries)
    for name in names[1:]:
        diffs = ", ".join(f"{m} {summaries[name][f'{m}_vs_baseline']['mean']:+.3f} "
                          f"[{summaries[name][f'{m}_vs_baseline']['ci95'][0]:+.2f},"
                          f"{summaries[name][f'{m}_vs_baseline']['ci95'][1]:+.2f}]"
                          for m in METRICS if summaries[name].get(f"{m}_vs_baseline"))
        print(f"  {name} vs {names[0]}: {diffs}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=100, help="questions in the sample (split equally by persona)")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--judge-model", default="gpt-4.1")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    conn = connect()
    store = ThreadStore(conn)
    retriever = HybridRetriever(DenseRetriever(conn), BM25.from_database(conn))
    pipelines = {name: RagPipeline(retriever, store, **CONFIGS[name]) for name in args.configs}
    judge = Judge(args.judge_model)

    cases = sample_cases(available_cases(load_cases(TEST_SET_PATH), store), args.n, args.seed)
    start = time.time()

    # Search once per question (one database connection, so this part runs in order)
    jobs = []
    for case in cases:
        retrieval = next(iter(pipelines.values())).retrieve(case["query"])
        for name, pipeline in pipelines.items():
            jobs.append((name, case, retrieval, pipeline.context(retrieval)))
    print(f"search + context for {len(cases)} questions: {time.time() - start:.0f}s", flush=True)

    # Answer and grade in parallel (OpenAI calls only)
    def run(job):
        name, case, retrieval, blocks = job
        answer = pipelines[name].generate(case["query"], blocks, retrieval)
        return name, grade(judge, case, answer)

    rows = {name: [] for name in args.configs}
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            name, row = future.result()
            rows[name].append(row)
            if done % 25 == 0:
                print(f"{done}/{len(jobs)} answers graded  {time.time() - start:.0f}s", flush=True)

    order = {c["query"]: i for i, c in enumerate(cases)}
    for name in rows:
        rows[name].sort(key=lambda r: order[r["query"]])
    baseline = args.configs[0]
    summaries = {name: summarise(rows[name], None if name == baseline else rows[baseline]) for name in args.configs}
    print_table(summaries)

    judge_tokens = [sum(r["judge"][m]["tokens"][i] for rs in rows.values() for r in rs for m in METRICS) for i in (0, 1)]
    answer_tokens = [sum(r["usage"][k] for rs in rows.values() for r in rs) for k in ("prompt_tokens", "completion_tokens")]
    print(f"\ntokens: answers ({get_settings().llm_model}) in {answer_tokens[0]:,} out {answer_tokens[1]:,} | "
          f"judge ({args.judge_model}) in {judge_tokens[0]:,} out {judge_tokens[1]:,} | {time.time() - start:.0f}s")

    REPORTS_DIR.mkdir(exist_ok=True)
    settings = asdict(get_settings())
    settings.pop("database_url")  # never write the connection string (it holds the password)
    report = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "code": git_version(),
        "settings": settings,
        "judge_model": args.judge_model,
        "configs": {name: CONFIGS[name] for name in args.configs},
        "sample": {"test_set": str(TEST_SET_PATH.relative_to(ROOT_DIR)),
                   "sha256": hashlib.sha256(TEST_SET_PATH.read_bytes()).hexdigest()[:16],
                   "n": len(cases), "seed": args.seed},
        "tokens": {"answers": answer_tokens, "judge": judge_tokens},
        "duration_s": round(time.time() - start),
        "summary": summaries,
        "rows": rows,
    }
    path = REPORTS_DIR / f"answer_eval_{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Wrote {path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
