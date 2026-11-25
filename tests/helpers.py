"""Small fakes and builders shared by the tests."""
from datetime import datetime
from types import SimpleNamespace


def answer(answer_id, score=0, accepted=False, year=2020, body="answer text"):
    return {"answer_id": answer_id, "score": score, "is_accepted": accepted,
            "creation_date": datetime(year, 1, 1), "body": body}


def thread(question_id, answers, title=None, body="question text"):
    return {"question": {"question_id": question_id, "title": title or f"Thread {question_id}", "body": body,
                         "score": 0, "creation_date": datetime(2020, 1, 1), "accepted_answer_id": None},
            "answers": answers}


class FakeStore:
    """Stands in for ThreadStore: returns threads from a dictionary."""
    def __init__(self, threads):
        self.threads = threads

    def get_threads(self, question_ids):
        return {qid: self.threads[qid] for qid in question_ids if qid in self.threads}


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, top_k=None):
        return list(self.hits)


class FakeLLM:
    """Stands in for the OpenAI client: records each call and returns a fixed reply."""
    def __init__(self, reply="An answer [1]."):
        self.reply = reply
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            model="fake-model",
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        )
