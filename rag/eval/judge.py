"""LLM judge for answer quality: the RAGAS metric definitions, plus our citation check.

    faithfulness       split the answer into claims; share supported by ANY of the given sources
                       (catches made-up facts)
    citation_accuracy  each cited sentence checked against ONLY the sources it cites
                       (catches right facts with wrong numbers; RAGAS has no such metric)
    answer_relevance   the judge writes 3 questions the answer would answer; mean similarity of
                       their embeddings to the real question (0 if the answer is noncommittal)
    reference_coverage share of the reference answer's key points that the answer covers

Implemented here instead of importing the ragas package, which would downgrade
the openai library and pull in LangChain. The judge is a stronger model than
the one that writes the answers, and its grades should be spot-checked by hand.
Each method returns a dict with "score" (None when not applicable), details,
and "tokens" (judge input, output) so the evaluation can report its own cost.
"""
import json
import re

import numpy as np
from openai import OpenAI

from rag.config import get_settings
from rag.generation.citations import CITATION_RE

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
SYSTEM = "You are a strict, careful grader of answers produced by a question-answering system. Reply only with JSON."


def _sources_text(sources) -> str:
    return "\n\n".join(f"[{s.n}]\n{s.body}" for s in sources)


class Judge:
    def __init__(self, model: str = "gpt-4.1", client: OpenAI | None = None):
        self.model = model
        self.client = client or OpenAI(timeout=120, max_retries=5)

    def _ask(self, prompt: str) -> tuple[dict, tuple[int, int]]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(response.choices[0].message.content), (response.usage.prompt_tokens,
                                                                 response.usage.completion_tokens)

    def faithfulness(self, answer: str, sources) -> dict:
        data, tokens = self._ask(f"""<sources>
{_sources_text(sources)}
</sources>

<answer>
{answer}
</answer>

List every factual claim made in the answer as a short standalone sentence (ignore citation markers like [3]).
Statements that the sources do not cover something are not claims. Example code counts as one claim:
"the answer shows code that does X". For each claim decide whether the sources state it or directly imply it.
Reply as JSON: {{"claims": [{{"claim": "...", "supported": true}}]}}""")
        claims = data.get("claims", [])
        score = sum(bool(c.get("supported")) for c in claims) / len(claims) if claims else None
        return {"score": score, "claims": claims, "tokens": tokens}

    def citation_accuracy(self, answer: str, sources) -> dict:
        by_number = {s.n: s for s in sources}
        items = []
        for sentence in SENTENCE_SPLIT.split(answer):
            numbers = sorted({int(n) for m in CITATION_RE.finditer(sentence)
                              for n in re.split(r"\s*,\s*", m.group(2)) if int(n) in by_number})
            if numbers:
                items.append((sentence.strip(), numbers))
        if not items:
            return {"score": None, "results": [], "tokens": (0, 0)}

        cited = [by_number[n] for n in sorted({n for _, ns in items for n in ns})]
        listing = "\n".join(f'{i}. (cites {", ".join(f"[{n}]" for n in ns)}) {s}' for i, (s, ns) in enumerate(items, 1))
        data, tokens = self._ask(f"""<sources>
{_sources_text(cited)}
</sources>

<sentences>
{listing}
</sentences>

For each numbered sentence, judge it ONLY against the sources it cites (not the other sources):
"yes" if those sources support everything the sentence states, "partly" if they support some of it,
"no" if they do not support it.
Reply as JSON: {{"results": [{{"i": 1, "verdict": "yes"}}]}}""")
        points = {"yes": 1.0, "partly": 0.5, "no": 0.0}
        verdicts = [r.get("verdict", "no") for r in data.get("results", [])]
        score = sum(points.get(v, 0.0) for v in verdicts) / len(verdicts) if verdicts else None
        return {"score": score, "sentences": [s for s, _ in items], "results": data.get("results", []),
                "tokens": tokens}

    def answer_relevance(self, question: str, answer: str) -> dict:
        data, tokens = self._ask(f"""<answer>
{answer}
</answer>

Write 3 different questions that this answer would be a good response to. Also say whether the answer is
noncommittal (it mostly says it cannot answer, or is vague).
Reply as JSON: {{"questions": ["...", "...", "..."], "noncommittal": false}}""")
        questions = [q for q in data.get("questions", []) if q][:3]
        if data.get("noncommittal") or not questions:
            return {"score": 0.0, "questions": questions, "noncommittal": True, "tokens": tokens}
        response = self.client.embeddings.create(model=get_settings().embedding_model, input=[question] + questions)
        vectors = np.array([d.embedding for d in response.data])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        score = float(np.mean(vectors[1:] @ vectors[0]))
        return {"score": score, "questions": questions, "noncommittal": False, "tokens": tokens}

    def reference_coverage(self, reference: str, answer: str) -> dict:
        data, tokens = self._ask(f"""<reference_answer>
{reference}
</reference_answer>

<answer>
{answer}
</answer>

List the key points of the reference answer (at most 6). For each, decide whether the answer covers it
(same meaning; wording may differ).
Reply as JSON: {{"points": [{{"point": "...", "covered": true}}]}}""")
        points = data.get("points", [])
        score = sum(bool(p.get("covered")) for p in points) / len(points) if points else None
        return {"score": score, "points": points, "tokens": tokens}
