import importlib
from bs4 import BeautifulSoup
import html
from openai import OpenAI
from dotenv import load_dotenv

cleanup_qa = importlib.import_module("02_cleanup_qa")
questions, answers = cleanup_qa.questions, cleanup_qa.answers

load_dotenv(override=True)
client = OpenAI()

def clean_body(raw_body):
    if not raw_body:
        return ""
    decoded = html.unescape(raw_body)
    soup = BeautifulSoup(decoded, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return text

question_records = []

for qid, q in questions.items():
    cleaned = clean_body(q["body"])
    record = {
        "question_id": qid,
        "title": q["title"],
        "tags": q["tags"],
        "score": q["score"],
        "view_count": q["view_count"],
        "answer_count": q["answer_count"],
        "accepted_answer_id": q["accepted_answer_id"],
        "creation_date": q["creation_date"],
        "last_activity_date": q["last_activity_date"],
        "body_clean": cleaned,
        "question_info": f"{q['title']} | Tags: {', '.join(q['tags'])} | {cleaned}",
    }
    question_records.append(record)

print(len(question_records), "question records built")
print(question_records["1"] if isinstance(question_records, dict) else next(r for r in question_records if r["question_id"] == "1"))