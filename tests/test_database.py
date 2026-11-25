"""Checks against the real database. Run with:  RUN_DB_TESTS=1 .venv/bin/python -m pytest -m db

No OpenAI calls: the query vector is a stored question embedding.
"""
import pytest

from rag.config import get_settings
from rag.db import connect
from rag.retrieval.dense import SEARCH_SQL

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def conn():
    connection = connect()
    yield connection
    connection.close()


def test_search_uses_the_hnsw_index(conn):
    # If someone changes <=> to another distance, Postgres silently reads every row (28 s instead of ~1 s)
    vector = conn.execute("SELECT embedding FROM questions LIMIT 1").fetchone()[0]
    plan = "\n".join(row[0] for row in conn.execute("EXPLAIN " + SEARCH_SQL, {"query": vector, "top_k": 10}))
    assert "questions_embedding_hnsw_idx" in plan


def test_connect_sets_ef_search(conn):
    assert conn.execute("SHOW hnsw.ef_search").fetchone()[0] == str(get_settings().hnsw_ef_search)


def test_every_answer_belongs_to_a_question(conn):
    orphans = conn.execute("""SELECT count(*) FROM answers a
                              LEFT JOIN questions q ON q.question_id = a.question_id
                              WHERE q.question_id IS NULL""").fetchone()[0]
    assert orphans == 0
