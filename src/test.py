from split_qa import questions, answers
from collections import Counter

answer_counts = Counter(q["answer_count"] for q in questions.values())
over_5 = sum(1 for q in questions.values() if int(q["answer_count"] or 0) > 5)
print(over_5, "questions have more than 5 answers")
print(f"{over_5 / len(questions) * 100:.1f}% of all questions")

answers_in_over5 = sum(int(q["answer_count"] or 0) for q in questions.values() if int(q["answer_count"] or 0) > 5)
print(answers_in_over5, "answers live inside the over-5 threads")
print(f"{answers_in_over5 / len(answers) * 100:.1f}% of all answers")
print(questions["1"])