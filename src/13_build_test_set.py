"""Build a 100-question retrieval test set with ground truth.

Threads are chosen (not at random) from 5 question-vote buckets, 20 each,
spread across tags. For each thread the gold answers are its top 3 answers by
the same quality score build_context uses (non-downvoted, accepted always kept).
An LLM writes a realistic user question that the thread answers, in different
words from the title; questions too close to the title or not answerable by
the gold answers are rejected.

Output: test_set.jsonl, one test case per line.
"""
import importlib
import json
import math
import random
import sqlite3
from pathlib import Path

import chromadb
import tiktoken
from dotenv import load_dotenv
from lxml import etree
from openai import OpenAI

SRC_DIR = Path(__file__).resolve().parent
RAW_DIR = SRC_DIR.parent / "data" / "raw" / "security-stackexchange"
TEST_SET = SRC_DIR / "test_set.jsonl"

load_dotenv(SRC_DIR.parent / ".env")
client = OpenAI()
MODEL = "gpt-4.1-mini"

ctx = importlib.import_module("12_build_context")
tokenize = importlib.import_module("07_bm25_index").tokenize
encoding = tiktoken.get_encoding("cl100k_base")

# Question-vote buckets from the vote percentiles (p40=1, p60=3, p80=6, p95=22).
BUCKETS = [
    ("high", 23, None),
    ("medium_high", 7, 22),
    ("medium", 4, 6),
    ("lower_medium", 2, 3),
    ("low", 0, 1),
]
PER_BUCKET = 20
GOLD_ANSWERS = 3
MIN_ANSWER_TOKENS, MAX_ANSWER_TOKENS = 50, 1500
MAX_TITLE_OVERLAP = 0.4
STYLES = ["short", "conversational", "scenario"]
SEED = 42


def quality(ans):
    """Same quality score as build_context step 2."""
    return (ctx.W_VOTES * math.log1p(max(ans["votes"], 0))
            + (ctx.W_ACCEPTED if ans["is_accepted"] else 0)
            + ctx.W_RECENCY * ctx.age_years(ans["creation_date"]))


def gold_answers(answers):
    """Top 3 by quality among non-downvoted answers; the accepted one is always kept."""
    eligible = sorted((a for a in answers if a["votes"] >= 0 or a["is_accepted"]),
                      key=quality, reverse=True)
    gold = eligible[:GOLD_ANSWERS]
    accepted = next((a for a in eligible if a["is_accepted"]), None)
    if accepted and accepted not in gold:
        gold = gold[:-1] + [accepted]
    return gold


def primary_tags():
    """question_id -> first tag (tags live in the Chroma questions metadata)."""
    col = chromadb.PersistentClient(path=str(SRC_DIR / "chroma_db")).get_collection("questions")
    tags, page = {}, 5000  # paged: one get() for all rows exceeds SQLite's variable limit
    for offset in range(0, col.count(), page):
        data = col.get(include=["metadatas"], limit=page, offset=offset)
        for qid, m in zip(data["ids"], data["metadatas"]):
            tags[qid] = (m.get("tags") or "").split(", ")[0]
    return tags


def duplicate_links():
    """question_id -> ids of questions linked to it as duplicates (either direction)."""
    links = {}
    for _, el in etree.iterparse(str(RAW_DIR / "PostLinks.xml"), events=("end",), tag="row"):
        if el.get("LinkTypeId") == "3":
            a, b = el.get("PostId"), el.get("RelatedPostId")
            links.setdefault(a, set()).add(b)
            links.setdefault(b, set()).add(a)
        el.clear()
    return links


def candidates_by_bucket():
    """Eligible threads per bucket, shuffled and spread across primary tags."""
    conn = sqlite3.connect(ctx.THREADS_DB)
    rows = conn.execute("SELECT question_id, votes FROM questions WHERE votes >= 0").fetchall()
    tags = primary_tags()
    rng = random.Random(SEED)
    buckets = {}
    for name, low, high in BUCKETS:
        ids = [qid for qid, v in rows if v >= low and (high is None or v <= high)]
        rng.shuffle(ids)
        # Round-robin over tags so one topic can't dominate a bucket.
        by_tag = {}
        for qid in ids:
            by_tag.setdefault(tags.get(qid, ""), []).append(qid)
        spread, queues = [], list(by_tag.values())
        rng.shuffle(queues)
        while queues:
            queues = [q for q in queues if q]
            for q in queues:
                spread.append(q.pop())
        buckets[name] = spread
    return buckets


def eligible_thread(thread):
    gold = gold_answers(thread["answers"])
    if not gold:
        return None
    top_len = len(encoding.encode(gold[0]["body"]))
    if not (MIN_ANSWER_TOKENS <= top_len <= MAX_ANSWER_TOKENS):
        return None
    return gold


def clip(text, tokens):
    ids = encoding.encode(text)
    return text if len(ids) <= tokens else encoding.decode(ids[:tokens]) + " ..."


def ask_json(system, user):
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return json.loads(response.choices[0].message.content)


GENERATE_SYSTEM = (
    "You write evaluation questions for a security Q&A search engine. Given a Stack "
    "Exchange thread, write the question a real user would type to find it. Use your "
    "own words: describe the problem the way someone who has not read the title would, "
    "rephrasing its wording (keep a technical term only if there is no natural "
    "alternative); do not mention the thread, "
    "IDs, or the answer itself. Then write a 2-3 sentence reference answer that "
    "summarises what the good answers say. Reply as JSON: "
    '{"question": "...", "reference_answer": "..."}'
)
STYLE_HINT = {
    "short": "Keep the question short, like a search query (under 12 words).",
    "conversational": "Write it conversationally, as one or two plain sentences.",
    "scenario": "Describe the user's situation or problem, then ask.",
}
CHECK_SYSTEM = (
    "You check evaluation data. Answer whether the given answers actually answer the "
    'question. Reply as JSON: {"answerable": true|false, "reason": "..."}'
)


def thread_prompt(question, gold):
    answers = "\n\n".join(f"[Answer, {a['votes']:+d} votes{', accepted' if a['is_accepted'] else ''}]\n"
                          f"{clip(a['body'], 700)}" for a in gold)
    return (f"Title: {question['title']}\n\nQuestion body:\n{clip(question['body'], 600)}"
            f"\n\nGood answers:\n{answers}")


def title_overlap(query, title):
    q = set(tokenize(query))
    return len(q & set(tokenize(title))) / max(len(q), 1)


def make_case(thread_id, thread, gold, style, links, in_corpus):
    q = thread["question"]
    context = thread_prompt(q, gold)
    for _ in range(3):
        out = ask_json(GENERATE_SYSTEM, f"{STYLE_HINT[style]}\n\n{context}")
        query = (out.get("question") or "").strip()
        if not query or title_overlap(query, q["title"]) > MAX_TITLE_OVERLAP:
            continue
        check = ask_json(CHECK_SYSTEM, f"Question: {query}\n\n{context}")
        if not check.get("answerable"):
            continue
        accepted = next((a["answer_id"] for a in thread["answers"] if a["is_accepted"]), None)
        return {
            "query": query,
            "style": style,
            "gold_thread_id": thread_id,
            "also_correct_threads": sorted((links.get(thread_id, set()) & in_corpus) - {thread_id}),
            "gold_answer_ids": [a["answer_id"] for a in gold],
            "accepted_answer_id": accepted,
            "reference_answer": out.get("reference_answer", ""),
            "question_votes": q["votes"],
            "thread_title": q["title"],
        }
    return None


def build_test_set():
    store = ctx.ThreadStore()
    in_corpus = {r[0] for r in store.conn.execute("SELECT question_id FROM questions")}
    links = duplicate_links()
    candidates = candidates_by_bucket()
    cases = []
    for name, _, _ in BUCKETS:
        made = 0
        for thread_id in candidates[name]:
            if made == PER_BUCKET:
                break
            thread = store.get_threads([thread_id])[thread_id]
            gold = eligible_thread(thread)
            if not gold:
                continue
            case = make_case(thread_id, thread, gold, STYLES[made % len(STYLES)], links, in_corpus)
            if case:
                cases.append({"bucket": name, **case})
                made += 1
                print(f"{name:<13} {made:>2}/{PER_BUCKET}  {case['query'][:80]}", flush=True)

    with open(TEST_SET, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")
    print(f"\nWrote {len(cases)} test cases -> {TEST_SET}")


if __name__ == "__main__":
    build_test_set()
