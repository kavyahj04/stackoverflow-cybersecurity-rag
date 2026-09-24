import importlib
import json
import chromadb
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SRC_DIR = Path(__file__).resolve().parent

parsing_answers = importlib.import_module("03_parsing_answers")
parsing_questions = importlib.import_module("04_parsing_questions")
answer_records = parsing_answers.answer_records
question_records = parsing_questions.question_records

client = chromadb.PersistentClient(path=str(SRC_DIR / "chroma_db"))
BATCH_SIZE = 500

def load_embeddings(path):
    embeddings = {}
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            embeddings[row["id"]] = row["embedding"]
    return embeddings

def build_metadata(record):
    metadata = {
        "score": int(record.get("score", 0)),
        "creation_date": record.get("creation_date", ""),
        "last_activity_date": record.get("last_activity_date", ""),
        "tags": ", ".join(record.get("tags", [])),
    }
    if record.get("view_count"):
        metadata["title"] = record.get("title", "")
        metadata["view_count"] = record.get("view_count", "")
        metadata["accepted_answer_id"] = int(record.get("accepted_answer_id") or 0)
    else:
        metadata["thread_id"] = record.get("thread_id", "")
        metadata["question_title"] = record.get("question_title", "")
        metadata["is_accepted"] = record.get("is_accepted", False)
    return metadata

def store_collection(name, records, id_field, text_field, embeddings_map):
    collection = client.get_or_create_collection(name)
    for i in range(0,len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        collection.add(
            ids = [r[id_field] for r in batch],
            embeddings=[embeddings_map[r[id_field]] for r in batch],
            documents=[r[text_field] for r in batch],
            metadatas=[build_metadata(r) for r in batch],
        )
    print(f"{name}: stored {i + len(batch)}/{len(records)}")

answer_embeddings = load_embeddings(str(SRC_DIR / "answer_embeddings.jsonl"))
question_embeddings = load_embeddings(str(SRC_DIR / "question_embeddings.jsonl"))

store_collection("answers", answer_records, "answer_id", "question_info", answer_embeddings)
store_collection("questions", question_records, "question_id", "question_info", question_embeddings)

