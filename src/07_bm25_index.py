import importlib
import pickle
import re
import os
from pathlib import Path
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS 

SRC_DIR = Path(__file__).resolve().parent
INDEX_PATH = str(SRC_DIR / "bm25_answers.pkl")
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    tokens = TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS] 


def build_index():
    parsing_answers = importlib.import_module("03_parsing_answers")
    answer_records = parsing_answers.answer_records
    ids = [r["answer_id"] for r in answer_records]
    corpus = [tokenize(r["question_info"]) for r in answer_records]
    bm25 = BM25Okapi(corpus)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids}, f)
    print(f"built BM25 index over {len(ids)} answers -> {INDEX_PATH}")
    return bm25, ids


def load_index():
    if not os.path.exists(INDEX_PATH):
        return build_index()
    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["ids"]


if __name__ == "__main__":
    build_index()
