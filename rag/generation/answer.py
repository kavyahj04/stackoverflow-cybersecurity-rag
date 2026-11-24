"""The full question -> cited answer pipeline.

    1. search          dense, plus BM25 when the question has an identifier (hybrid)
    2. relevance       nothing close enough and no identifier match -> "not found", no LLM call
    3. context         best answers per thread, trimmed to the token budget
    4. prompt          numbered sources + rules
    5. LLM             OpenAI, with timeout and retries
    6. citations       remove numbers that don't exist; the Sources list is built by code, not the LLM
"""
import time
from dataclasses import dataclass, field

from openai import OpenAI

from rag.config import get_settings
from rag.context import build_context, fit_budget
from rag.generation.citations import CitationReport, check_citations
from rag.generation.prompt import Source, build_messages, number_sources

NOT_FOUND = "I couldn't find discussions on Security Stack Exchange that answer this question."


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Source] = field(default_factory=list)   # only the cited ones, by number
    found: bool = True                                    # False: nothing relevant, LLM not called
    retrieval: dict = field(default_factory=dict)         # what search returned, for logs and debugging
    citations: CitationReport = field(default_factory=CitationReport)
    usage: dict = field(default_factory=dict)             # model, tokens, seconds

    def render(self) -> str:
        """The text a researcher sees: the answer, then the Sources list."""
        if not self.sources:
            return self.text
        lines = [self.text, "", "Sources:"]
        for s in self.sources:
            flags = ", ".join((["accepted"] if s.is_accepted else []) + [f"{s.score:+d} votes", str(s.year)])
            lines.append(f"[{s.n}] {s.thread_title} ({flags})\n    {s.url}")
        return "\n".join(lines)


class RagPipeline:
    def __init__(self, retriever, store, client: OpenAI | None = None):
        settings = get_settings()
        self.retriever = retriever      # DenseRetriever or HybridRetriever
        self.store = store              # ThreadStore
        self.client = client or OpenAI(timeout=settings.llm_timeout_s, max_retries=2)

    def ask(self, question: str) -> Answer:
        settings = get_settings()
        start = time.time()

        # 1-2. Search, and stop early when nothing is close enough
        hits = self.retriever.search(question)
        distances = [h.distance for h in hits if h.distance is not None]
        best_distance = min(distances) if distances else None
        exact_match = any(h.source == "exact" for h in hits)
        retrieval = {
            "threads": [h.question_id for h in hits],
            "exact_match": exact_match,
            "best_distance": round(best_distance, 4) if best_distance is not None else None,
        }
        if not exact_match and (best_distance is None or best_distance > settings.max_distance):
            return Answer(question, NOT_FOUND, found=False, retrieval=retrieval,
                          usage={"seconds": round(time.time() - start, 2)})

        # 3-4. Context within budget, numbered sources, prompt
        blocks = fit_budget(build_context(retrieval["threads"], self.store))
        retrieval["threads_in_context"] = len(blocks)
        sources = number_sources(blocks)
        messages = build_messages(question, blocks, sources)

        # 5. LLM call
        options = {"temperature": settings.llm_temperature} if settings.llm_temperature is not None else {}
        response = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            max_completion_tokens=settings.max_answer_tokens,
            **options,
        )
        choice = response.choices[0]

        # 6. Citation check; the Sources list comes from our table, so links are always real
        text, report = check_citations(choice.message.content or "", len(sources))
        by_number = {s.n: s for s in sources}
        return Answer(
            question=question,
            text=text,
            sources=[by_number[n] for n in sorted(report.cited)],
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
