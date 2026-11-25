"""BM25 keyword search over questions (title + body).

BM25 scores a question for a query by adding, for every query word it contains:

    idf(word) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average_length))

    idf   rare words count more: a CVE id in 2 of 57,141 questions is a strong
          signal, "security" in thousands is almost none
    tf    how often the word appears in that question; the k1 term makes the
          gain flatten, so 10 mentions are not 10 times better than 1
    b     long questions contain more words by chance, so their score is
          scaled down by their length

The tokenizer keeps security identifiers (CVE-2017-7494, MS12-020, CWE-79,
0x80070005) as single tokens. A normal word splitter would break CVE-2017-7494
into "cve", "2017", "7494", and "2017" matches every question that mentions
the year.

The index is built from Postgres when the program starts (about 10 seconds),
so it always matches the database and no index file can go out of date.

Memory matters because the server runs on a small machine. Positions are kept
as plain 4-byte numbers (a Python int takes 28 bytes) in one big array per kind
instead of 106k small ones, and questions are read in batches, not all at once.
"""
import math
import re
from array import array
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
import snowballstemmer

# Identifiers first: at each position the regex tries these before plain words
EXACT_PATTERNS = [
    r"\bcve-\d{4}-\d{4,7}\b",   # CVE-2017-7494
    r"\bms\d{2}-\d{3}\b",       # Microsoft bulletin MS12-020
    r"\bcwe-\d{1,4}\b",         # weakness type CWE-79
    r"\b0x[0-9a-f]{6,16}\b",    # error codes / addresses 0x80070005
]
EXACT_RE = re.compile("|".join(EXACT_PATTERNS))
TOKEN_RE = re.compile("|".join(EXACT_PATTERNS + [r"[a-z0-9]+"]))

# Function words only (content words like "system" or "call" matter here)
STOP_WORDS = frozenset("""
a about above after again against all am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for
from further had has have having he her here hers herself him himself his how i if in
into is it its itself just me more most my myself no nor not now of off on once only or
other our ours ourselves out over own same she should so some such than that the their
theirs them themselves then there these they this those through to too under until up
very was we were what when where which while who whom why will with would you your
yours yourself yourselves s t d ll m re ve
""".split())

_stemmer = snowballstemmer.stemmer("english")


@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    """9 million words but only ~100k different ones, so each is stemmed once.

    The limit keeps the cache from growing forever on new words in user queries."""
    return _stemmer.stemWord(word)


def is_exact(token: str) -> bool:
    return EXACT_RE.fullmatch(token) is not None


def tokenize(text: str) -> list[str]:
    """Lowercase words, stemmed (exploits -> exploit), stop words dropped; identifiers kept whole."""
    tokens = []
    for token in TOKEN_RE.findall(text.lower()):
        if is_exact(token):
            tokens.append(token)
        elif token not in STOP_WORDS:
            tokens.append(stem(token))
    return tokens


class BM25:
    def __init__(self, question_ids, documents, k1: float = 1.2, b: float = 0.75):
        """documents[i] is the token list of question question_ids[i].

        documents may be a generator (read once, in order); question_ids may be a list that the
        generator fills while it runs. Both are only used after every document has been read.
        """
        self.k1 = k1
        lengths = array("f")
        # Inverted index: word -> (which questions contain it, how many times in each)
        postings = defaultdict(lambda: (array("i"), array("f")))
        for i, doc in enumerate(documents):
            lengths.append(len(doc))
            for token, count in Counter(doc).items():
                docs, counts = postings[token]
                docs.append(i)
                counts.append(count)

        self.question_ids = np.array(question_ids)
        lengths = np.frombuffer(lengths, dtype=np.float32)
        # The length part of the formula depends only on the question, so compute it once
        self.length_norm = k1 * (1 - b + b * lengths / lengths.mean())

        # All postings in two big arrays; word number j owns positions offsets[j]:offsets[j+1]
        n = len(lengths)
        self.word_number = {token: j for j, token in enumerate(postings)}
        sizes = np.array([len(docs) for docs, _ in postings.values()], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(sizes)])
        self.docs = np.empty(self.offsets[-1], dtype=np.int32)
        self.tf = np.empty(self.offsets[-1], dtype=np.float32)
        for j, (docs, counts) in enumerate(postings.values()):
            self.docs[self.offsets[j]:self.offsets[j + 1]] = np.frombuffer(docs, dtype=np.int32)
            self.tf[self.offsets[j]:self.offsets[j + 1]] = np.frombuffer(counts, dtype=np.float32)
        # This idf form is always positive, even for words in more than half the questions
        self.idf = {token: math.log(1 + (n - int(size) + 0.5) / (int(size) + 0.5))
                    for token, size in zip(postings, sizes)}

    @classmethod
    def from_database(cls, conn, batch_size: int = 2_000) -> "BM25":
        """Reads the questions in batches through a server-side cursor, so they are never all in memory."""
        question_ids = []

        def documents(cursor):
            for qid, title, body in cursor:
                question_ids.append(qid)
                yield tokenize(f"{title} {body}")

        with conn.transaction(), conn.cursor(name="bm25_build") as cursor:
            cursor.itersize = batch_size
            cursor.execute("SELECT question_id, title, body FROM questions ORDER BY question_id")
            return cls(question_ids, documents(cursor))

    def __contains__(self, token: str) -> bool:
        return token in self.word_number

    def postings(self, token: str) -> tuple[np.ndarray, np.ndarray]:
        """(positions of the questions containing the word, count in each)."""
        j = self.word_number[token]
        start, end = self.offsets[j], self.offsets[j + 1]
        return self.docs[start:end], self.tf[start:end]

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        """BM25 score of every question for the query (0 for questions sharing no word with it)."""
        scores = np.zeros(len(self.question_ids), dtype=np.float32)
        for token in set(query_tokens):
            if token in self:
                docs, tf = self.postings(token)
                scores[docs] += self.idf[token] * tf * (self.k1 + 1) / (tf + self.length_norm[docs])
        return scores

    def containing(self, tokens: list[str]) -> np.ndarray:
        """Positions of the questions that contain at least one of these tokens."""
        return np.unique(np.concatenate([self.postings(t)[0] for t in tokens]))
