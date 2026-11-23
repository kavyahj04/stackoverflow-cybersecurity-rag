"""Read whole threads (a question plus all its answers) from Postgres."""

QUESTIONS_SQL = """
    SELECT question_id, title, body, score, creation_date, accepted_answer_id
    FROM questions
    WHERE question_id = ANY(%s)
"""
# ORDER BY makes the answer order (and so tie-breaks in answer selection) repeatable
ANSWERS_SQL = """
    SELECT answer_id, question_id, body, score, creation_date, is_accepted
    FROM answers
    WHERE question_id = ANY(%s)
    ORDER BY answer_id
"""


class ThreadStore:
    def __init__(self, conn):
        self.conn = conn  # from rag.db.connect()

    def get_threads(self, question_ids: list[int]) -> dict[int, dict]:
        """question_id -> {"question": {...}, "answers": [{...}, ...]}. Two queries in total."""
        ids = list(question_ids)
        threads = {}
        for qid, title, body, score, created, accepted_id in self.conn.execute(QUESTIONS_SQL, (ids,)):
            threads[qid] = {
                "question": {
                    "question_id": qid,
                    "title": title,
                    "body": body,
                    "score": score,
                    "creation_date": created,
                    "accepted_answer_id": accepted_id,
                },
                "answers": [],
            }
        for aid, qid, body, score, created, is_accepted in self.conn.execute(ANSWERS_SQL, (ids,)):
            threads[qid]["answers"].append({
                "answer_id": aid,
                "body": body,
                "score": score,
                "creation_date": created,
                "is_accepted": is_accepted,
            })
        return threads

    def existing_ids(self, question_ids) -> set[int]:
        """Which of these question ids are in the database."""
        rows = self.conn.execute("SELECT question_id FROM questions WHERE question_id = ANY(%s)",
                                 (list(question_ids),))
        return {row[0] for row in rows}
