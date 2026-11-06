import importlib

split_qa = importlib.import_module("01_split_qa")
questions, answers = split_qa.questions, split_qa.answers

# Close reasons we don't search: 102 and 2 = off-topic, 103 = unclear
REMOVED_CLOSE_REASONS = {"102", "2", "103"}

answered_question_ids = {a["parent_id"] for a in answers}

questions = {
    qid: q for qid, q in questions.items()
    if qid in answered_question_ids and q["close_reason"] not in REMOVED_CLOSE_REASONS
}
answers = [a for a in answers if a["parent_id"] in questions]

print(f"Final: {len(questions)} questions, {len(answers)} answers")
