"""What goes in and out of the API, as typed models.

FastAPI uses these to reject bad input before any of our code runs (empty or
too-long questions), and to publish the interactive docs at /docs, so a
researcher can also call the API from a script or notebook.
"""
import re

from pydantic import BaseModel, Field, field_validator

from rag.config import get_settings
from rag.generation.answer import Answer
from rag.generation.prompt import SITE_URL, Source


def _clean_question(text: str) -> str:
    text = text.strip()
    if len(text) < 3:
        raise ValueError("Please write a question of at least 3 characters.")
    limit = get_settings().max_question_chars
    if len(text) > limit:
        raise ValueError(f"Please keep the question under {limit} characters.")
    return text


# --- requests -------------------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(examples=["How do I prevent SQL injection in PHP?"])

    @field_validator("question")
    @classmethod
    def check(cls, value: str) -> str:
        return _clean_question(value)


class SearchRequest(BaseModel):
    query: str = Field(examples=["CVE-2017-7494"])

    @field_validator("query")
    @classmethod
    def check(cls, value: str) -> str:
        return _clean_question(value)


# --- answer ---------------------------------------------------------------------------------------

class SourceOut(BaseModel):
    n: int                  # the number cited in the answer text, [n]
    answer_id: int
    question_id: int
    thread_title: str
    url: str                # link to the answer
    thread_url: str         # link to the whole thread
    votes: int
    accepted: bool
    year: int
    text: str               # the answer as the LLM read it, so the claim can be checked here


class RetrievalOut(BaseModel):
    exact_match: bool               # an identifier like CVE-2017-7494 matched (BM25 was used)
    best_distance: float | None     # cosine distance of the closest thread (0 = same meaning)
    threads_searched: int
    threads_in_context: int


class CitationsOut(BaseModel):
    invalid_removed: list[int]      # numbers the LLM made up, removed from the text
    uncited_sentences: int          # longer sentences without a citation


class UsageOut(BaseModel):
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cut_off: bool = False           # the answer hit the length limit
    seconds: float | None = None


class AnswerOut(BaseModel):
    question: str
    found: bool                     # False: nothing relevant in the data, the LLM was not called
    text: str                       # the answer, with [n] citations
    age_note: str | None
    sources: list[SourceOut]        # the cited sources, by number
    context_sources: list[SourceOut]  # everything the LLM was given
    retrieval: RetrievalOut
    citations: CitationsOut
    usage: UsageOut


def source_out(s: Source) -> SourceOut:
    return SourceOut(n=s.n, answer_id=s.answer_id, question_id=s.question_id, thread_title=s.thread_title,
                     url=s.url, thread_url=f"{SITE_URL}/q/{s.question_id}", votes=s.score,
                     accepted=s.is_accepted, year=s.year, text=s.body)


def answer_out(a: Answer) -> AnswerOut:
    r = a.retrieval
    return AnswerOut(
        question=a.question,
        found=a.found,
        text=a.text,
        age_note=a.age_note,
        sources=[source_out(s) for s in a.sources],
        context_sources=[source_out(s) for s in a.context_sources],
        retrieval=RetrievalOut(exact_match=r.get("exact_match", False), best_distance=r.get("best_distance"),
                               threads_searched=len(r.get("threads", [])),
                               threads_in_context=r.get("threads_in_context", 0)),
        citations=CitationsOut(invalid_removed=a.citations.invalid,
                               uncited_sentences=a.citations.uncited_sentences),
        usage=UsageOut(**a.usage),
    )


# --- search only ----------------------------------------------------------------------------------

class ThreadOut(BaseModel):
    question_id: int
    title: str
    url: str
    excerpt: str
    votes: int
    answers: int
    has_accepted: bool
    year: int
    match: str                      # "exact" (identifier found by BM25) or "dense" (similar meaning)
    distance: float | None
    relevant: bool                  # exact match, or close enough that /api/ask would answer from it


class SearchOut(BaseModel):
    query: str
    threads: list[ThreadOut]


def excerpt(text: str, length: int = 280) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= length else text[:length].rsplit(" ", 1)[0] + " …"


def thread_out(hit, thread: dict) -> ThreadOut:
    q = thread["question"]
    return ThreadOut(
        question_id=hit.question_id,
        title=q["title"],
        url=f"{SITE_URL}/q/{hit.question_id}",
        excerpt=excerpt(q["body"]),
        votes=q["score"],
        answers=len(thread["answers"]),
        has_accepted=any(a["is_accepted"] for a in thread["answers"]),
        year=q["creation_date"].year,
        match=hit.source,
        distance=round(hit.distance, 4) if hit.distance is not None else None,
        relevant=hit.source == "exact" or (hit.distance is not None
                                           and hit.distance <= get_settings().max_distance),
    )


# --- service ---------------------------------------------------------------------------------------

class InfoOut(BaseModel):
    threads: int                    # questions searchable
    llm_model: str
    embedding_model: str
    context: str                    # what the LLM is given per question
    max_question_chars: int
