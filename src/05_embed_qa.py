import importlib
import json
import os
import tiktoken
from pathlib import Path
from openai import OpenAI

SRC_DIR = Path(__file__).resolve().parent

parsing_answers = importlib.import_module("03_parsing_answers")
parsing_questions = importlib.import_module("04_parsing_questions")
answer_records = parsing_answers.answer_records
question_records = parsing_questions.question_records


client = OpenAI()
BATCH_SIZE = 500
MAX_TOKENS = 8191
encoding = tiktoken.encoding_for_model("text-embedding-3-large")

def truncate(text):
    tokens = encoding.encode(text)
    if len(tokens) <= MAX_TOKENS:
        return text
    return encoding.decode(tokens[:MAX_TOKENS])

def embed_batch(texts):
    texts = [truncate(t) for t in texts]
    response = client.embeddings.create(input=texts, model="text-embedding-3-large")
    return [item.embedding for item in response.data]

def embed_all(records, text_field, id_field, output_path):
    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                done_ids.add(json.loads(line)["id"])

    remaining = [r for r in records if r[id_field] not in done_ids]

    with open(output_path, "a") as f:
        for i in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[i : i + BATCH_SIZE]
            texts = [r[text_field] for r in batch]
            embeddings = embed_batch(texts)
            for record, embedding in zip(batch, embeddings):
                f.write(json.dumps({"id": record[id_field], "embedding": embedding}) + "\n")

embed_all(answer_records, "question_info", "answer_id", str(SRC_DIR / "answer_embeddings.jsonl"))
embed_all(question_records, "question_info", "question_id", str(SRC_DIR / "question_embeddings.jsonl"))