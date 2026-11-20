import importlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector import HalfVector
from pgvector.psycopg import register_vector

load_dotenv(override=True)

SRC_DIR = Path(__file__).resolve().parent
BATCH_SIZE = 1000

question_records = importlib.import_module("04_parsing_questions").question_records
answer_records = importlib.import_module("03_parsing_answers").answer_records


def load_embeddings(path):
    embeddings = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            # float16 = 2 bytes per number, the same size halfvec stores
            embeddings[row["id"]] = np.array(row["embedding"], dtype=np.float16)
    return embeddings


def to_int(value):
    return int(value) if value not in (None, "") else None


def to_time(value):
    return datetime.fromisoformat(value) if value else None


def question_row(r, embedding):
    return (
        int(r["question_id"]), r["title"], r["body_clean"], r["tags"],
        int(r["score"]), to_int(r["view_count"]), to_int(r["accepted_answer_id"]),
        to_time(r["creation_date"]), to_time(r["last_activity_date"]),
        HalfVector(embedding),
    )


def answer_row(r):
    return (
        int(r["answer_id"]), int(r["thread_id"]), r["answer_body_clean"],
        int(r["score"]), r["is_accepted"],
        to_time(r["creation_date"]), to_time(r["last_activity_date"]),
    )


QUESTION_SQL = """
    INSERT INTO questions (question_id, title, body, tags, score, view_count,
                           accepted_answer_id, creation_date, last_activity_date, embedding)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""
ANSWER_SQL = """
    INSERT INTO answers (answer_id, question_id, body, score, is_accepted,
                         creation_date, last_activity_date)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

# Check before touching the database, so a missing embedding changes nothing
embeddings = load_embeddings(SRC_DIR / "question_embeddings.jsonl")
missing = [r["question_id"] for r in question_records if r["question_id"] not in embeddings]
if missing:
    raise ValueError(f"{len(missing)} questions have no embedding, so nothing was changed")

# Everything inside this block is one transaction: it all saves at the end,
# or, if anything fails, nothing is saved and the old data stays.
# psycopg is a library that lets Python talk to Postgres.
with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute("TRUNCATE questions, answers")

        for i in range(0, len(question_records), BATCH_SIZE):
            batch = question_records[i : i + BATCH_SIZE]
            cur.executemany(QUESTION_SQL, [question_row(r, embeddings[r["question_id"]]) for r in batch])
            print(f"questions: {i + len(batch)}/{len(question_records)}", flush=True)

        for i in range(0, len(answer_records), BATCH_SIZE):
            batch = answer_records[i : i + BATCH_SIZE]
            cur.executemany(ANSWER_SQL, [answer_row(r) for r in batch])
            print(f"answers: {i + len(batch)}/{len(answer_records)}", flush=True)

print("Done: all rows saved")
