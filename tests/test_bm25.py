import math

from rag.retrieval.bm25 import BM25, is_exact, tokenize


def test_identifiers_stay_whole():
    tokens = tokenize("Is CVE-2017-7494 exploitable? See MS12-020, CWE-79 and error 0x80070005.")
    for identifier in ("cve-2017-7494", "ms12-020", "cwe-79", "0x80070005"):
        assert identifier in tokens
    assert "2017" not in tokens and "7494" not in tokens  # not split into pieces


def test_stop_words_dropped_and_words_stemmed():
    assert tokenize("How do I patch CVE-2021-44228 on my servers?") == ["patch", "cve-2021-44228", "server"]
    assert tokenize("exploits exploited exploiting") == ["exploit"] * 3


def test_is_exact():
    assert is_exact("cve-2017-7494")
    assert not is_exact("samba")
    assert not is_exact("cve-2017")


def test_scores_match_hand_calculation():
    docs = [["cve-1", "exploit"], ["exploit", "exploit", "samba", "x", "y"], ["samba"]]
    bm25 = BM25([1, 2, 3], docs)
    n, avg, k1, b = 3, 8 / 3, 1.2, 0.75

    def by_hand(token, doc):
        df = sum(token in d for d in docs)
        tf = doc.count(token)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avg))

    query = ["exploit", "samba"]
    expected = [sum(by_hand(t, d) for t in query) for d in docs]
    assert [round(float(s), 5) for s in bm25.scores(query)] == [round(e, 5) for e in expected]


def test_rare_words_count_more():
    bm25 = BM25([1, 2, 3], [["common", "rare"], ["common"], ["common"]])
    assert bm25.idf["rare"] > bm25.idf["common"] > 0  # idf stays positive even for very common words


def test_repeats_level_off():
    once = BM25([1, 2], [["word", "a", "b"], ["c", "d", "e"]]).scores(["word"])[0]
    many = BM25([1, 2], [["word"] * 10 + ["a", "b"], ["c"] * 12]).scores(["word"])[0]
    assert once < many < 10 * once


def test_longer_documents_score_lower():
    bm25 = BM25([1, 2], [["word", "a"], ["word"] + ["filler"] * 20])
    short, long = bm25.scores(["word"])
    assert short > long


def test_unknown_words_score_zero_and_containing():
    bm25 = BM25([10, 20, 30], [["cve-1", "a"], ["b"], ["cve-1", "cve-2"]])
    assert list(bm25.scores(["nothing"])) == [0, 0, 0]
    assert "cve-1" in bm25 and "cve-9" not in bm25
    assert bm25.containing(["cve-1", "cve-2"]).tolist() == [0, 2]  # positions, each once
