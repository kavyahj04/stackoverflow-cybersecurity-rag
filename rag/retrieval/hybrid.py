"""Triggered hybrid search: BM25 only when the query contains an exact identifier.

    query has no identifier found in the data  -> dense search only (unchanged)
    query has one (e.g. CVE-2017-7494)         -> questions containing it, ranked by
                                                  BM25 on the whole query, come first;
                                                  dense results fill the remaining places

Dense search understands meaning but not identifiers: on bare CVE ids it found
the right thread only about 60% of the time. BM25 alone would hurt normal
questions (mixing it into every query lowered recall before), so it only runs
when it can help.

search() returns (question_id, source) pairs, best first; source is "exact" or
"dense", so a report can show where each result came from.
"""
import numpy as np

from rag.config import get_settings
from rag.retrieval.bm25 import BM25, is_exact, tokenize
from rag.retrieval.dense import DenseRetriever


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, bm25: BM25):
        self.dense = dense
        self.bm25 = bm25

    def exact_terms(self, query: str) -> list[str]:
        """Identifiers in the query that occur somewhere in our questions (the trigger)."""
        return sorted({t for t in tokenize(query) if is_exact(t) and t in self.bm25})

    def search(self, query: str, top_k: int | None = None) -> list[tuple[int, str]]:
        top_k = top_k or get_settings().top_k
        dense_ids = [qid for qid, _ in self.dense.search(query, top_k)]
        terms = self.exact_terms(query)
        if not terms:
            return [(qid, "dense") for qid in dense_ids]

        # Only questions that contain the identifier; BM25 orders them using every query word
        candidates = self.bm25.containing(terms)
        scores = self.bm25.scores(tokenize(query))[candidates]
        best = candidates[np.argsort(-scores, kind="stable")][:top_k]
        exact_ids = [int(qid) for qid in self.bm25.question_ids[best]]

        rest = [qid for qid in dense_ids if qid not in exact_ids]
        return [(qid, "exact") for qid in exact_ids] + [(qid, "dense") for qid in rest][: top_k - len(exact_ids)]
