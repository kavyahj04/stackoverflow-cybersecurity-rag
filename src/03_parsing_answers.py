import importlib

cleanup_text = importlib.import_module("03_text_cleaning")
cleanup_qa = importlib.import_module("02_cleanup_qa")
questions, answers = cleanup_qa.questions, cleanup_qa.answers


answer_records = []

for ans in answers:
    parent = questions[ans["parent_id"]]
    cleaned = cleanup_text.clean_body(ans["body"])
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

