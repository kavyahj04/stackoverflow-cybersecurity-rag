from bs4 import BeautifulSoup
import html
from split_qa import questions, answers
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
client = OpenAI()

def clean_body(raw_body):
    if not raw_body:
        return ""
    decoded = html.unescape(raw_body)
    soup = BeautifulSoup(decoded, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return text

answer_records = []

for ans in answers:
    parent = questions[ans["parent_id"]]
    cleaned = clean_body(ans["body"])
    record = {
        "answer_id" : ans["id"],
        "thread_id" : ans["parent_id"],
        "question_title": parent["title"],
        "tags" : parent["tags"],
        "score" : ans["score"],
        "is_accepted" : parent["accepted_answer_id"] == ans["id"],
        "creation_date" : ans["creation_date"],
        "last_activity_date" : ans["last_activity_date"],
        "answer_body_clean" : cleaned,
        "question_info": f"Question Title: {parent['title']} | Tags: {', '.join(parent['tags'])} | Answer: {cleaned}"
    }
    answer_records.append(record)
print(len(answer_records), "answers records built")
print(answer_records[0])
