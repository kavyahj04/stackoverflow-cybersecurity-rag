from rag.retrieval.bm25 import BM25, tokenize
from rag.retrieval.dense import Hit
from rag.retrieval.hybrid import HybridRetriever

from helpers import FakeRetriever

DENSE = [Hit(qid, 0.3 + i / 100, "dense") for i, qid in enumerate([101, 102, 103, 104, 105])]
QUESTIONS = {
    201: "CVE-2017-7494 Samba remote code execution exploit",
    202: "How does the CVE-2017-7494 SambaCry bug work",
    103: "Is my Samba share affected by CVE-2017-7494",   # also returned by dense
    204: "Unrelated question about TLS certificates",
}


def retriever():
    bm25 = BM25(list(QUESTIONS), [tokenize(text) for text in QUESTIONS.values()])
    return HybridRetriever(FakeRetriever(DENSE), bm25)


def test_no_identifier_means_dense_only():
    assert retriever().search("How do I prevent SQL injection?", top_k=5) == DENSE


def test_identifier_not_in_our_data_means_dense_only():
    assert retriever().search("What is CVE-2099-12345?", top_k=5) == DENSE


def test_exact_matches_first_then_dense_fills_the_rest():
    hits = retriever().search("Is CVE-2017-7494 exploitable?", top_k=5)
    assert [h.source for h in hits] == ["exact"] * 3 + ["dense"] * 2
    assert {h.question_id for h in hits[:3]} == {201, 202, 103}
    assert [h.question_id for h in hits[3:]] == [101, 102]          # dense order kept
    assert len(hits) == 5 and len({h.question_id for h in hits}) == 5  # total is top_k, no duplicates


def test_thread_found_by_both_keeps_its_dense_distance():
    hits = {h.question_id: h for h in retriever().search("CVE-2017-7494", top_k=5)}
    assert hits[103].source == "exact" and hits[103].distance == DENSE[2].distance
    assert hits[201].distance is None  # dense never returned it


def test_more_exact_matches_than_places():
    hits = retriever().search("CVE-2017-7494 samba", top_k=2)
    assert [h.source for h in hits] == ["exact", "exact"]
