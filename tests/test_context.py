from datetime import datetime

from rag.context import build_context, fit_budget, pick_answers, quality

from helpers import FakeStore, answer, thread

NOW = datetime(2026, 11, 24)


def test_accepted_bonus_beats_double_votes():
    # Thread 71696: the accepted answer (7 votes) ranks above a 14-vote answer
    assert quality(answer(1, score=7, accepted=True, year=2014), NOW) > quality(answer(2, score=14, year=2014), NOW)


def test_downvoted_dropped_unless_accepted():
    picked = pick_answers([answer(1, score=-4), answer(2, score=3), answer(3, score=-1, accepted=True)], 3, NOW)
    assert [a["answer_id"] for a in picked] == [3, 2]


def test_accepted_always_kept_even_when_weak():
    answers = [answer(i, score=100, year=2025) for i in range(1, 4)] + [answer(9, score=0, accepted=True, year=2009)]
    picked = pick_answers(answers, 2, NOW)
    assert len(picked) == 2 and picked[-1]["answer_id"] == 9


def test_build_context_keeps_search_order_and_original_ranks():
    store = FakeStore({1: thread(1, [answer(10)]), 3: thread(3, [answer(30)])})
    blocks = build_context([3, 2, 1], store, max_threads=10, max_answers=2, now=NOW)  # thread 2 is missing
    assert [(b["question_id"], b["rank"]) for b in blocks] == [(3, 1), (1, 3)]


def test_fit_budget_drops_from_the_bottom_and_keeps_the_top_thread():
    big = "word " * 400
    store = FakeStore({i: thread(i, [answer(i * 10, body=big)]) for i in (1, 2, 3)})
    blocks = build_context([1, 2, 3], store, max_threads=10, max_answers=2, now=NOW)
    assert [b["question_id"] for b in fit_budget(blocks, max_tokens=900)] == [1, 2]
    assert [b["question_id"] for b in fit_budget(blocks, max_tokens=10)] == [1]  # never empty
