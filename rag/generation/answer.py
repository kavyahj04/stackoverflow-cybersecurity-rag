"""The full question -> cited answer pipeline.

    1. search          dense, plus BM25 when the question has an identifier (hybrid)
    2. relevance       nothing close enough and no identifier match -> "not found", no LLM call
    3. context         best answers per thread, trimmed to the token budget
    4. prompt          numbered sources + rules (+ two worked examples)
    5. LLM             OpenAI, with timeout and retries
    6. citations       remove numbers that don't exist; the Sources list is built by code, not the LLM
    7. age note        when most cited answers are old, a fixed note says so (the LLM ignored this rule)

retrieve(), context() and generate() are separate so the evaluation can search
once and try several context sizes or prompts on the same results.
"""
import time
from dataclasses import dataclass, field
from datetime import date

from openai import OpenAI

from rag.config import get_settings
from rag.context import build_context, fit_budget
from rag.generation.citations import CitationReport, check_citations
from rag.generation.prompt import Source, build_messages, number_sources

NOT_FOUND = "I couldn't find discussions on Security Stack Exchange that answer this question."


def age_note(cited: list[Source], today: date | None = None) -> str | None:
    """A fixed warning when at least half of the cited answers are old."""
    if not cited:
        return None
    year = (today or date.today()).year
    old = sorted(s.year for s in cited if year - s.year >= get_settings().old_advice_years)
    if len(old) * 2 < len(cited):
        return None
    span = str(old[0]) if old[0] == old[-1] else f"{old[0]}-{old[-1]}"
    return (f"Note: {len(old)} of the {len(cited)} cited answers are from {span}. "
            "Security advice may have changed since then.")


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Source] = field(default_factory=list)          # only the cited ones, by number
    context_sources: list[Source] = field(default_factory=list)  # everything the LLM was given
    age_note: str | None = None                                  # set when most cited answers are old
    found: bool = True                                           # False: nothing relevant, LLM not called
    retrieval: dict = field(default_factory=dict)                # what search returned, for logs and debugging
    citations: CitationReport = field(default_factory=CitationReport)
    usage: dict = field(default_factory=dict)                    # model, tokens, seconds

    def render(self) -> str:
        """The text a researcher sees: the answer, then the Sources list."""
        if not self.sources:
            return self.text
        lines = [self.text, ""] + ([self.age_note, ""] if self.age_note else []) + ["Sources:"]
        for s in self.sources:
            flags = ", ".join((["accepted"] if s.is_accepted else []) + [f"{s.score:+d} votes", str(s.year)])
            lines.append(f"[{s.n}] {s.thread_title} ({flags})\n    {s.url}")
        return "\n".join(lines)


class RagPipeline:
    def __init__(self, retriever, store, client: OpenAI | None = None, few_shot: bool | None = None,
                 max_threads: int | None = None, max_answers: int | None = None):
        settings = get_settings()
        self.retriever = retriever      # DenseRetriever or HybridRetriever
        self.store = store              # ThreadStore
        self.client = client or OpenAI(timeout=settings.llm_timeout_s, max_retries=2)
        self.few_shot = settings.prompt_few_shot if few_shot is None else few_shot
        self.max_threads = max_threads or settings.context_threads
        self.max_answers = max_answers or settings.max_answers_per_thread

    def retrieve(self, question: str) -> dict:
        """Search, and decide whether anything is close enough to answer from."""
        hits = self.retriever.search(question)
        distances = [h.distance for h in hits if h.distance is not None]
        best = min(distances) if distances else None
        exact_match = any(h.source == "exact" for h in hits)
        return {
            "threads": [h.question_id for h in hits],
            "exact_match": exact_match,
            "best_distance": round(best, 4) if best is not None else None,
            "found": exact_match or (best is not None and best <= get_settings().max_distance),
        }

    def context(self, retrieval: dict) -> list[dict]:
        blocks = build_context(retrieval["threads"], self.store, self.max_threads, self.max_answers)
        return fit_budget(blocks)

    def generate(self, question: str, blocks: list[dict], retrieval: dict | None = None) -> Answer:
        settings = get_settings()
        start = time.time()
        retrieval = dict(retrieval or {}, threads_in_context=len(blocks))
        sources = number_sources(blocks)
        messages = build_messages(question, blocks, sources, few_shot=self.few_shot)

        options = {"temperature": settings.llm_temperature} if settings.llm_temperature is not None else {}
        response = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            max_completion_tokens=settings.max_answer_tokens,
            **options,
        )
        choice = response.choices[0]

        # The Sources list comes from our table, so links are always real
        text, report = check_citations(choice.message.content or "", len(sources))
        by_number = {s.n: s for s in sources}
        cited = [by_number[n] for n in sorted(report.cited)]
        return Answer(
            question=question,
            text=text,
            sources=cited,
            context_sources=sources,
            age_note=age_note(cited),
            retrieval=retrieval,
            citations=report,
            usage={
                "model": response.model,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "cut_off": choice.finish_reason == "length",   # answer hit max_answer_tokens
                "seconds": round(time.time() - start, 2),
            },
        )

    def ask(self, question: str) -> Answer:
        start = time.time()
        retrieval = self.retrieve(question)
        if not retrieval["found"]:
            return Answer(question, NOT_FOUND, found=False, retrieval=retrieval,
                          usage={"seconds": round(time.time() - start, 2)})
        answer = self.generate(question, self.context(retrieval), retrieval)
        answer.usage["seconds"] = round(time.time() - start, 2)
        return answer
