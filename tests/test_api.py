"""The API with fake services: no database and no OpenAI, so these run in a second."""
from contextlib import contextmanager

import openai
import psycopg
import pytest
from fastapi.testclient import TestClient

from rag.api.app import RateLimiter, create_app
from rag.generation.answer import NOT_FOUND, RagPipeline
from rag.retrieval.dense import Hit

from helpers import FakeLLM, FakeRetriever, FakeStore, answer, thread

STORE = FakeStore({1: thread(1, [answer(11, score=5, accepted=True, year=2024), answer(12, score=2, year=2012)],
                             title="Preventing SQL injection"),
                   2: thread(2, [answer(21, year=2024)])})


class FakeServices:
    threads = 2

    def __init__(self, hits=None, llm=None):
        self.hits = hits or [Hit(1, 0.30, "dense"), Hit(2, 0.40, "dense")]
        self.llm = llm or FakeLLM("Use prepared statements [1]. Old answers suggested escaping [2][9].")
        self.borrowed = False   # is a database connection out right now

    @contextmanager
    def pipeline(self):
        self.borrowed = True
        try:
            yield RagPipeline(FakeRetriever(self.hits), STORE, client=self.llm)
        finally:
            self.borrowed = False

    def ping(self):
        return True

    def close(self):
        pass


def client_for(services):
    return TestClient(create_app(lambda: services), raise_server_exceptions=False)


def test_ask_returns_cited_answer_with_sources():
    with client_for(FakeServices()) as client:
        body = client.post("/api/ask", json={"question": "  How do I prevent SQL injection?  "}).json()
    assert body["found"] and body["question"] == "How do I prevent SQL injection?"
    assert body["text"] == "Use prepared statements [1]. Old answers suggested escaping [2]."
    assert [s["n"] for s in body["sources"]] == [1, 2] and len(body["context_sources"]) == 3
    first = body["sources"][0]
    assert first["url"] == "https://security.stackexchange.com/a/11" and first["accepted"]
    assert first["thread_url"] == "https://security.stackexchange.com/q/1" and first["text"] == "answer text"
    assert body["citations"]["invalid_removed"] == [9]
    assert body["usage"]["prompt_tokens"] == 100 and body["usage"]["seconds"] is not None


def test_database_connection_is_returned_before_the_llm_call():
    services = FakeServices()
    services.llm.chat.completions.create = lambda **kw: pytest.fail("connection still borrowed") \
        if services.borrowed else FakeLLM()._create(**kw)
    with client_for(services) as client:
        assert client.post("/api/ask", json={"question": "SQL injection?"}).status_code == 200


def test_nothing_relevant_means_no_llm_call():
    services = FakeServices(hits=[Hit(1, 0.9, "dense")])
    with client_for(services) as client:
        body = client.post("/api/ask", json={"question": "best cookie recipe?"}).json()
    assert not body["found"] and body["text"] == NOT_FOUND and services.llm.calls == []


@pytest.mark.parametrize("question", ["", "  a ", "x" * 1001])
def test_bad_questions_rejected_with_a_readable_message(question):
    with client_for(FakeServices()) as client:
        response = client.post("/api/ask", json={"question": question})
    assert response.status_code == 422 and isinstance(response.json()["detail"], str)


def test_search_returns_threads_without_calling_the_llm():
    services = FakeServices(hits=[Hit(2, None, "exact"), Hit(1, 0.3, "dense")])
    with client_for(services) as client:
        threads = client.post("/api/search", json={"query": "CVE-2017-7494"}).json()["threads"]
    assert [(t["question_id"], t["match"], t["relevant"]) for t in threads] == [(2, "exact", True), (1, "dense", True)]
    assert threads[1]["has_accepted"] and threads[1]["answers"] == 2 and services.llm.calls == []


def test_out_of_credits_gives_a_clear_503():
    services = FakeServices()

    def no_credits(**kwargs):
        raise openai.RateLimitError("quota", response=_response(429), body={"type": "insufficient_quota", "code": "credit_balance_exhausted"})  # as sent in Nov 2026
    services.llm.chat.completions.create = no_credits
    with client_for(services) as client:
        response = client.post("/api/ask", json={"question": "SQL injection?"})
    assert response.status_code == 503 and "credits" in response.json()["detail"]


def test_database_down_gives_503():
    services = FakeServices()

    @contextmanager
    def broken():
        raise psycopg.OperationalError("connection refused")
        yield
    services.pipeline = broken
    with client_for(services) as client:
        response = client.post("/api/ask", json={"question": "SQL injection?"})
    assert response.status_code == 503 and "connection refused" not in response.text  # no internals leaked


def test_rate_limit():
    limiter = RateLimiter(per_minute=2)
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a")
    assert limiter.allow("b")  # counted per address


def test_info_and_health():
    with client_for(FakeServices()) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        assert client.get("/api/info").json()["threads"] == 2


def _response(status):
    import httpx
    return httpx.Response(status, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
