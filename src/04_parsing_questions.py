import importlib

cleanup_text = importlib.import_module("03_01_text_cleaning")
cleanup_qa = importlib.import_module("02_cleanup_qa")
questions, answers = cleanup_qa.questions, cleanup_qa.answers


question_records = []

for qid, q in questions.items():
    cleaned = cleanup_text.clean_body(q["body"])
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
        "embedded_question": f"{q['title']} | Tags: {', '.join(q['tags'])} | {cleaned}",
    }
    question_records.append(record)

print(len(question_records), "question records built")