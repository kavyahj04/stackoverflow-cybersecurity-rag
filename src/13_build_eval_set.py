"""Build a retrieval test set from Stack Exchange duplicate links.

When a question is closed as a duplicate, PostLinks.xml links it (LinkTypeId=3)
to the original. The duplicate's title is a real, differently-worded version of
the original question, so:
    query = duplicate's title
    gold  = original thread(s) (+ their accepted answers, for a stricter check)

Only pairs where the duplicate is NOT in the cleaned corpus are kept, so search
can't succeed by finding the duplicate itself.
"""
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from lxml import etree

SRC_DIR = Path(__file__).resolve().parent
RAW_DIR = SRC_DIR.parent / "data" / "raw" / "security-stackexchange"
THREADS_DB = SRC_DIR / "threads.db"
EVAL_SET = SRC_DIR / "eval_set.jsonl"

DUPLICATE_LINK = "3"


def load_duplicate_links():
    """duplicate question id -> set of original question ids"""
    originals = defaultdict(set)
    for _, el in etree.iterparse(str(RAW_DIR / "PostLinks.xml"), events=("end",), tag="row"):
        if el.get("LinkTypeId") == DUPLICATE_LINK:
            originals[el.get("PostId")].add(el.get("RelatedPostId"))
        el.clear()
    return originals


def load_raw_titles(question_ids):
    """Titles from the raw dump (duplicates are usually not in the cleaned data)."""
    titles = {}
    for _, el in etree.iterparse(str(RAW_DIR / "Posts.xml"), events=("end",), tag="row"):
        if el.get("PostTypeId") == "1" and el.get("Id") in question_ids:
            titles[el.get("Id")] = el.get("Title")
        el.clear()
    return titles


def build_eval_set():
    conn = sqlite3.connect(THREADS_DB)
    accepted_by_thread = dict(conn.execute("SELECT question_id, accepted_answer_id FROM questions"))
    in_corpus = set(accepted_by_thread)

    links = load_duplicate_links()
    titles = load_raw_titles(set(links))

    cases = []
    for dup_id, original_ids in links.items():
        gold = sorted(original_ids & in_corpus)
        if dup_id in in_corpus or not gold or not titles.get(dup_id):
            continue
        cases.append({
            "query_id": dup_id,
            "query": titles[dup_id],
            "gold_thread_ids": gold,
            "gold_accepted_ids": [accepted_by_thread[t] for t in gold if accepted_by_thread[t]],
        })

    with open(EVAL_SET, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")
    print(f"duplicate links: {len(links)} | usable test cases: {len(cases)} -> {EVAL_SET}")
    return cases


if __name__ == "__main__":
    build_eval_set()
