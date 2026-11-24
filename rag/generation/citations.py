"""Level-1 citation check: plain code, free, runs on every answer.

Catches citation numbers that don't exist (the LLM cited [12] but there were
9 sources) and removes them, so a researcher never sees a broken reference.
It also counts sentences without any citation, a rough signal of claims the
researcher can't verify. Whether [2] really supports its sentence needs an LLM
judge; that belongs to the answer evaluation, not to every request.
"""
import re
from dataclasses import dataclass, field

CITATION_RE = re.compile(r"( ?)\[(\d+(?:\s*,\s*\d+)*)\]")   # [3] or [2, 5], with the space before it
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class CitationReport:
    cited: list[int] = field(default_factory=list)      # valid numbers, in first-use order
    invalid: list[int] = field(default_factory=list)    # numbers that don't exist (removed)
    uncited_sentences: int = 0


def check_citations(text: str, n_sources: int) -> tuple[str, CitationReport]:
    report = CitationReport()

    def keep_valid(match):
        numbers = [int(x) for x in re.split(r"\s*,\s*", match.group(2))]
        valid = [n for n in numbers if 1 <= n <= n_sources]
        report.invalid += [n for n in numbers if n not in valid]
        for n in valid:
            if n not in report.cited:
                report.cited.append(n)
        # Nothing valid left: drop the space too, so no double space remains
        return match.group(1) + "".join(f"[{n}]" for n in valid) if valid else ""

    clean = CITATION_RE.sub(keep_valid, text)
    sentences = [s for s in SENTENCE_END.split(clean) if len(s.split()) >= 6]  # skip short lines/headings
    report.uncited_sentences = sum(1 for s in sentences if not CITATION_RE.search(s))
    return clean, report
