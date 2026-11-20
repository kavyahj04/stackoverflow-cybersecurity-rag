"""Read threads (a question plus all its answers) from Postgres.

Replaces the SQLite ThreadStore in 12_build_context.py. get_threads returns
the same shape as before (ids and dates as text, score as "votes"), so
context.py and evaluate.py work without other changes.
"""
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

SRC_DIR = Path(__file__).resolve().parent.parent
load_dotenv(SRC_DIR.parent / ".env", override=True)

QUESTIONS_SQL = """
    SELECT question_id, title, body, score, creation_date, accepted_answer_id
    FROM questions
    WHERE question_id = ANY(%s)
"""
ANSWERS_SQL = """
    SELECT answer_id, question_id, body, score, creation_date, is_accepted
    FROM answers
    WHERE question_id = ANY(%s)
"""


def as_text(value):
    return str(value) if value is not None else None


class PostgresThreadStore:
    def __init__(self):
        self.conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)

    def get_threads(self, thread_ids):
        """thread_id -> {"question": {...}, "answers": [{...}]}"""
        ids = [int(t) for t in thread_ids]
        threads = {}

        for qid, title, body, score, created, accepted_id in self.conn.execute(QUESTIONS_SQL, (ids,)):
            threads[str(qid)] = {
                "question": {
                    "title": title,
                    "body": body,
                    "votes": score,
                    "creation_date": created.isoformat(),
                    "accepted_answer_id": as_text(accepted_id),
                },
                "answers": [],
            }

        for aid, qid, body, score, created, is_accepted in self.conn.execute(ANSWERS_SQL, (ids,)):
            threads[str(qid)]["answers"].append({
                "answer_id": str(aid),
                "body": body,
                "votes": score,
                "creation_date": created.isoformat(),
                "is_accepted": is_accepted,
            })

        return threads
