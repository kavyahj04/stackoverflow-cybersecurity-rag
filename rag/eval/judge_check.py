"""Check the LLM judge against a person (blind).

    make    picks items from an answer-eval report and writes a sheet without the
            judge's verdicts, plus a separate key file
    score   reads your verdicts from the sheet and reports how often you and the
            judge agree

The judge's numbers are only worth trusting if it agrees with a careful human
most of the time. The sheet hides its verdicts so you can't be swayed by them.
Unsupported claims and "partly"/"no" citations are rare, so they are
over-sampled: those are the verdicts that move the scores.

Usage (from rag-dev):
    python -m rag.eval.judge_check make  [--report reports/answer_eval_....json] [--config v2_7x2]
    python -m rag.eval.judge_check score reports/judge_check_<time>.md
"""
import argparse
import glob
import json
import random
import re
from datetime import datetime
from pathlib import Path

from rag.config import REPORTS_DIR, ROOT_DIR
from rag.db import connect
from rag.eval.answer_eval import CONFIGS
from rag.generation.answer import RagPipeline
from rag.generation.prompt import number_sources
from rag.retrieval.bm25 import BM25
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.store import ThreadStore

WORD = re.compile(r"[a-z0-9]+")


def overlap(a: str, b: str) -> int:
    return len(set(WORD.findall(a.lower())) & set(WORD.findall(b.lower())))


def quote(text: str, limit: int = 1500) -> str:
    text = text if len(text) <= limit else text[:limit] + " ..."
    return "\n".join("> " + line for line in text.splitlines())


def make(args):
    report_path = Path(args.report or sorted(glob.glob(str(REPORTS_DIR / "answer_eval_*.json")))[-1])
    report = json.loads(report_path.read_text())
    rows = report["rows"][args.config]
    rng = random.Random(args.seed)

    # Candidate items, split into the rare verdicts and the common ones
    cite_rare, cite_common, claim_rare, claim_common = [], [], [], []
    for row in rows:
        cit = row["judge"]["citation_accuracy"]
        for i, (sentence, result) in enumerate(zip(cit.get("sentences", []), cit.get("results", []))):
            item = {"kind": "citation", "row": row, "sentence": sentence, "judge": result.get("verdict")}
            (cite_rare if item["judge"] != "yes" else cite_common).append(item)
        for claim in row["judge"]["faithfulness"]["claims"]:
            item = {"kind": "claim", "row": row, "claim": claim["claim"], "judge": "yes" if claim.get("supported") else "no"}
            (claim_rare if item["judge"] == "no" else claim_common).append(item)
    half = args.n // 2
    pick = lambda rare, common, k: (rng.sample(rare, min(len(rare), k // 2))
                                    + rng.sample(common, k - min(len(rare), k // 2)))
    items = pick(cite_rare, cite_common, half) + pick(claim_rare, claim_common, args.n - half)
    rng.shuffle(items)

    # Rebuild each answer's sources from the threads search returned. Newer reports store them;
    # for older ones, take them from the hybrid retrieval report (same retriever, same results),
    # so no embedding call is needed. Context building is deterministic for a given config.
    conn = connect()
    pipeline = RagPipeline(None, ThreadStore(conn), **CONFIGS[args.config])
    saved = {}
    retrieval_reports = sorted(glob.glob(str(REPORTS_DIR / "retrieval_eval_hybrid_test_set_human_*.json")))
    if retrieval_reports:
        saved = {c["query"]: c["retrieved"] for c in json.loads(Path(retrieval_reports[-1]).read_text())["cases"]}
    sources_for = {}
    for item in items:
        query = item["row"]["query"]
        if query not in sources_for:
            threads = item["row"].get("threads") or saved.get(query)
            if threads is None:  # last resort: search again (needs the OpenAI API)
                pipeline.retriever = HybridRetriever(DenseRetriever(conn), BM25.from_database(conn))
                threads = pipeline.retrieve(query)["threads"]
            sources_for[query] = number_sources(pipeline.context({"threads": threads}))

    lines = [f"# Judge check ({len(items)} items)", "",
             f"From `{report_path.relative_to(ROOT_DIR)}`, config `{args.config}`.",
             "For each item, write your verdict after **Your verdict:**. Judge it yourself; the judge's",
             "verdicts are in a separate file.", ""]
    key = []
    for i, item in enumerate(items, 1):
        sources = {s.n: s for s in sources_for[item["row"]["query"]]}
        lines += [f"---", "", f"## Item {i}", "", f"**Question:** {item['row']['query']}", ""]
        if item["kind"] == "citation":
            numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", item["sentence"]) if int(n) in sources})
            lines += ["**Do the cited sources support everything this sentence says?** (yes / partly / no)", "",
                      f"**Sentence:** {item['sentence']}", ""]
            shown = [sources[n] for n in numbers]
        else:
            lines += ["**Is this claim supported by the sources?** (yes / no)", "",
                      f"**Claim:** {item['claim']}", "",
                      "_The 3 sources sharing the most words with the claim are shown; all sources of this answer "
                      "are in the report if you need them._", ""]
            shown = sorted(sources.values(), key=lambda s: -overlap(item["claim"], s.body))[:3]
        for s in shown:
            lines += [f"**Source [{s.n}]** ({s.url})", "", quote(s.body), ""]
        lines += ["**Your verdict:** ", ""]
        key.append({"item": i, "kind": item["kind"], "judge": item["judge"]})

    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    sheet = REPORTS_DIR / f"judge_check_{stamp}.md"
    sheet.write_text("\n".join(lines))
    (REPORTS_DIR / f"judge_check_{stamp}.key.json").write_text(json.dumps(key, indent=2))
    print(f"Wrote {sheet.relative_to(ROOT_DIR)} (fill it in) and its key file")


def score(args):
    sheet = Path(args.sheet)
    key = json.loads(sheet.with_suffix(".key.json").read_text())
    mine = re.findall(r"\*\*Your verdict:\*\*[ \t]*(\S*)", sheet.read_text())
    pairs = [(k, v.strip().lower().strip(".")) for k, v in zip(key, mine) if v.strip()]
    if not pairs:
        print("No verdicts filled in yet.")
        return
    agree = [k["judge"] == v for k, v in pairs]
    print(f"Agreement: {sum(agree)}/{len(agree)} = {sum(agree) / len(agree):.0%}")
    for kind in ("citation", "claim"):
        sub = [a for (k, _), a in zip(pairs, agree) if k["kind"] == kind]
        if sub:
            print(f"  {kind:<9}{sum(sub)}/{len(sub)}")
    for (k, v), a in zip(pairs, agree):
        if not a:
            print(f"  item {k['item']}: you said {v!r}, the judge said {k['judge']!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("make")
    m.add_argument("--report")
    m.add_argument("--config", default="v2_7x2", choices=list(CONFIGS))
    m.add_argument("--n", type=int, default=20)
    m.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("score")
    s.add_argument("sheet")
    args = parser.parse_args()
    make(args) if args.command == "make" else score(args)


if __name__ == "__main__":
    main()
