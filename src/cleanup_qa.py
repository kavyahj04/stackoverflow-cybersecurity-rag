from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from split_qa import questions, answers

#  Handle questions with zero answers 

question_ids = set(questions.keys())
answered_question_ids = {a['parent_id'] for a in answers}

zero_answer_ids = question_ids - answered_question_ids
print(f"{len(zero_answer_ids)} questions have zero answers, dropping these")

questions = {qid: q for qid, q in questions.items() if qid not in zero_answer_ids}
answers = [a for a in answers if a['parent_id'] not in zero_answer_ids]
print(f"{len(questions)} questions remain after dropping zero-answer questions")

# Duplicate question check
# Global comparison across all remaining questions, no grouping,
# since duplicate questions can appear anywhere in the set.

def question_text(q):
    title = q.get('title', '') or ''
    body = q.get('body', '') or ''
    return f"{title} {body}"

ids = list(questions.keys())
texts = [question_text(questions[qid]) for qid in ids]

# Count answers per question, used to decide which duplicate to keep
answer_counts = defaultdict(int)
for a in answers:
    answer_counts[a['parent_id']] += 1

vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(texts)
sim_matrix = cosine_similarity(vectors)

threshold = 0.90  # slightly looser than the answer threshold, questions
                   # tend to be shorter and more prone to paraphrasing
to_drop_question_ids = set()

for i in range(len(ids)):
    if ids[i] in to_drop_question_ids:
        continue
    for j in range(i + 1, len(ids)):
        if ids[j] in to_drop_question_ids:
            continue
        if sim_matrix[i, j] >= threshold:
            # keep whichever has more answers, drop the other
            if answer_counts[ids[i]] >= answer_counts[ids[j]]:
                to_drop_question_ids.add(ids[j])
            else:
                to_drop_question_ids.add(ids[i])
                break

print(f"Found {len(to_drop_question_ids)} duplicate questions to drop")

questions = {qid: q for qid, q in questions.items() if qid not in to_drop_question_ids}
answers = [a for a in answers if a['parent_id'] not in to_drop_question_ids]
print(f"{len(questions)} questions remain after duplicate-question removal")

#Duplicate answer check, within each remaining question

answers_by_question = defaultdict(list)
for ans in answers:
    answers_by_question[ans['parent_id']].append(ans)

answer_threshold = 0.92
kept_answers = []
dropped_answer_count = 0

for parent_id, group in answers_by_question.items():
    if len(group) == 1:
        kept_answers.append(group[0])
        continue

    group_texts = [a['body'] for a in group]
    group_vectors = vectorizer.fit_transform(group_texts)
    group_sim = cosine_similarity(group_vectors)

    to_drop_local = set()
    for i in range(len(group)):
        if i in to_drop_local:
            continue
        for j in range(i + 1, len(group)):
            if j in to_drop_local:
                continue
            if group_sim[i, j] >= answer_threshold:
                score_i = int(group[i]['score'])
                score_j = int(group[j]['score'])
                if score_i >= score_j:
                    to_drop_local.add(j)
                else:
                    to_drop_local.add(i)
                    break

    for idx, ans in enumerate(group):
        if idx not in to_drop_local:
            kept_answers.append(ans)
        else:
            dropped_answer_count += 1

answers = kept_answers
print(f"Dropped {dropped_answer_count} near-duplicate answers")

# Final size check

final_question_count = len(questions)
final_answer_count = len(answers)
print(f"Final subset: {final_question_count} questions, {final_answer_count} answers")

if 1000 <= final_question_count <= 5000:
    print("Size looks reasonable, low thousands, as expected.")
elif final_question_count < 1000:
    print("Smaller than expected, worth checking whether too much got dropped.")
else:
    print("Larger than expected for a 'low thousands' target.")