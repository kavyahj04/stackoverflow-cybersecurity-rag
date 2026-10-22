import os
import pickle
import re
from functools import lru_cache
from pathlib import Path

import chromadb
import snowballstemmer
from rank_bm25 import BM25Okapi

SRC_DIR = Path(__file__).resolve().parent
INDEX_PATH = str(SRC_DIR / "bm25_index.pkl")
TOKEN_RE = re.compile(r"[a-z0-9]+")
PAGE = 5000  # Chroma get() page size (one call for everything exceeds SQLite's variable limit)
ANSWERS_PER_QUESTION_HIT = 3  # answers of a matched question that sparse search may return

# Function words only. sklearn's ENGLISH_STOP_WORDS also drops content words that
# matter here ("system", "call", "find", "full", "back", "show", ...).
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


@lru_cache(maxsize=None)
def _stem(token):
    return _stemmer.stemWord(token)


def tokenize(text):
    tokens = TOKEN_RE.findall(text.lower())
    return [_stem(t) for t in tokens if t not in STOP_WORDS]


def _read_collection(collection, with_metadata=False):
    ids, docs, metas = [], [], []
    include = ["documents", "metadatas"] if with_metadata else ["documents"]
    for offset in range(0, collection.count(), PAGE):
        page = collection.get(include=include, limit=PAGE, offset=offset)
        ids += page["ids"]
        docs += page["documents"]
        if with_metadata:
            metas += page["metadatas"]
    return ids, docs, metas


def build_index():
    """BM25 over answers (question_info) and over questions, built from Chroma's texts."""
    client = chromadb.PersistentClient(path=str(SRC_DIR / "chroma_db"))

    ids, docs, metas = _read_collection(client.get_collection("answers"), with_metadata=True)
    bm25 = BM25Okapi([tokenize(d) for d in docs])

    # thread -> its answers, highest score first (used when a question matches)
    thread_answers = {}
    for aid, meta in sorted(zip(ids, metas), key=lambda x: -x[1]["score"]):
        thread_answers.setdefault(meta["thread_id"], []).append(aid)

    q_ids, q_docs, _ = _read_collection(client.get_collection("questions"))
    q_bm25 = BM25Okapi([tokenize(d) for d in q_docs])

    index = {"bm25": bm25, "ids": ids, "q_bm25": q_bm25, "q_ids": q_ids,
             "thread_answers": thread_answers}
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    print(f"built BM25 over {len(ids)} answers and {len(q_ids)} questions -> {INDEX_PATH}")
    return index


@lru_cache(maxsize=1)
def _load():
    if not os.path.exists(INDEX_PATH):
        return build_index()
    with open(INDEX_PATH, "rb") as f:
        return pickle.load(f)


def load_index():
    """Answer index: (bm25, answer_ids)."""
    index = _load()
    return index["bm25"], index["ids"]


def load_question_index():
    """Question index: (bm25, question_ids, thread_id -> answer_ids by score)."""
    index = _load()
    return index["q_bm25"], index["q_ids"], index["thread_answers"]


if __name__ == "__main__":
    build_index()
