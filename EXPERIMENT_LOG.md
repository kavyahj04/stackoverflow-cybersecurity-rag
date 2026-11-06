# Experiment log: Security Stack Exchange RAG

A record of what was tried, what it scored, and what went wrong.

Dates are git commit dates, except the 2026-10-16 test runs, which use file timestamps.

**Reading the scores:** R@10 means the correct thread appeared in the top 10 results. MRR is higher when the correct thread is ranked nearer the top (1.0 is perfect).

---

## Latest progress: 2026-11-05

### What to say
> Today I reviewed my retrieval pipeline from the start, one file at a time, checking each step against real data. I finished the first two files. I added a close reason to each question and now filter out off-topic and unclear threads, which leaves 57,141 questions and 113,091 answers. I also removed my TF-IDF duplicate detection, because it was deleting unrelated questions and caught only 4 of about 2,800 real duplicates.
>
> I also found a bug in my HTML cleaning that deletes quoted code like `<script>`, which matters on a security site. It affects about 5,300 posts. The fix is tested but not applied yet. The biggest finding is about evaluation: the same answer search scored 1.00, 0.84 and 0.53 on three different test sets, and only the 0.53 test used queries written by real people. So before generation, I need to re-test on real queries.

### Done
- `01_split_qa.py`: added `close_reason`, removed debug code.
- `02_cleanup_qa.py`: replaced with the no-answer rule and close-reason filter; duplicate steps removed. Run and confirmed.
- This log created from git history and saved results.

### Pick up here tomorrow
1. `03_parsing_answers.py` is fixed (2026-11-06). Only its unused imports on lines 2-6 remain.
2. Continue file by file: `04`, `05`, `06`, and on to the evaluation files.
3. Don't run `05` or `06` yet. Chroma, `threads.db` and the BM25 index still hold the old 60,577-question data.

### Still undecided
- Whether to filter code 4 (not a real question). It isn't filtered at the moment.
- The production plan (Postgres with pgvector, and an API) is on hold until this review is finished.

---

## The main lesson so far

The same answer search scored very differently depending on the test:

| Test set | Who wrote the queries | Queries | R@10 | MRR |
|---|---|---|---|---|
| `test_set.jsonl` | LLM that saw the question and the good answers | 100 | **1.00** | 0.918 |
| `test_set_human.jsonl` | LLM that saw only the question, using 5 personas | 500 | **0.844** | 0.678 |
| `eval_set.jsonl` (now deleted) | Real people: titles of duplicate questions | 1,181 | **0.530** | 0.317 |

A score only means something alongside the test that produced it.

---

## Timeline

| Date | What was tried | Test | Result | Note |
|---|---|---|---|---|
| 09-03 | Split `Posts.xml`, clean tags, remove duplicates | Printed counts | Runs | Thresholds 0.90 and 0.92 chosen without a test |
| 09-08 | `Posts.xml` on disk was re-saved | Compared with the archive on 2026-11-05 | Line endings changed; invisible separators removed from 7 rows; no rows added or lost | Harmless, but raw data shouldn't be edited |
| 09-24 | Dense search (`text-embedding-3-small`) and BM25 | Hand-written questions | No recall score found | Choices made before anything was measured |
| 09-26 | RRF fusion of dense and BM25 | Hand-written questions | No recall score found | |
| 09-30 | Cross-encoder (MiniLM-L-6), keeping dense top 5 | None | No score | |
| 10-01 | MiniLM-L-12; context of 5 threads with 3, then 5 answers | Read by eye | No score | |
| 10-01 | Switched to `text-embedding-3-large`, added cleanup | None | No score | No before/after comparison, so the benefit is unknown |
| 10-09 | `bge-reranker-v2-m3` (MiniLM cut answers at 512 tokens), 10 threads | Duplicate titles, see below | See below | **First real measurement** |
| 10-16 | Deleted the duplicate test; LLM test set that saw answers | `test_set.jsonl` | R@10 1.00 | Too good to be true |
| 10-16 | LLM test set that saw only questions | `test_set_human.jsonl` | R@10 0.844 | Harder, but still LLM-written |
| 10-16 | BM25 tuning | `test_set_human.jsonl` | See below | Adding questions helped far more than stemming |
| 10-22 | Reranker on a sample, then on all queries | `test_set_human.jsonl` | Lowered R@10 | Cause not established |
| 10-23 | Fusion variants, question search | `test_set_human.jsonl` | Question search scored best | Led to the switch to question search |
| 10-23 | Question pipeline, end to end | `test_set_human.jsonl` | R@10 0.910 | Two of its metrics can't fail |

---

## Results

### Duplicate-title test (10-09), answer search only
Recover the scripts with `git show b40927b^:src/13_build_eval_set.py` and `git show b40927b^:src/14_evaluate_retrieval.py`.

| Method | Queries | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| Dense (answers) | 1,181 | 0.353 | 0.419 | 0.530 | 0.317 |
| BM25 (answers) | 1,181 | 0.158 | 0.202 | 0.262 | 0.149 |
| Dense (answers) | 150 | 0.340 | 0.400 | 0.507 | 0.305 |
| BM25 (answers) | 150 | 0.120 | 0.167 | 0.233 | 0.128 |
| Hybrid + reranker | 150 | 0.340 | 0.400 | 0.527 | 0.305 |
| `build_context` | 150 | 0.240 | 0.373 | 0.493 | 0.223 |

Question search was never run on this test.

### BM25 tuning (`test_set_human.jsonl`, 500 queries)
| Setup | R@10 | MRR |
|---|---|---|
| Answers only, baseline | 0.410 | 0.257 |
| Answers only, with stemming | 0.412 | 0.262 |
| Answers and questions, with stemming | 0.598 | 0.409 |

### Reranker (`test_set_human.jsonl`)
| Queries | Method | R@3 | R@5 | R@10 | R@20 | MRR |
|---|---|---|---|---|---|---|
| 100 | Dense | 0.60 | 0.65 | 0.72 | 0.80 | Not calculated |
| 100 | Hybrid + reranker | 0.59 | 0.68 | 0.72 | 0.76 | Not calculated |
| 500 | Dense | 0.730 | 0.788 | 0.844 | 0.904 | 0.678 |
| 500 | Hybrid + reranker | 0.726 | 0.778 | 0.808 | 0.840 | 0.670 |

On all 500 queries: moved into the top 10 for 9, out of the top 10 for 27, about 25 seconds per query.

The 100-query sample scored 0.72 where all 500 scored 0.844, which shows how far a small sample can move a score.

### Fusion variants (`test_set_human.jsonl`, 500 queries)
| Method | R@3 | R@5 | R@10 | R@20 | MRR |
|---|---|---|---|---|---|
| Dense on answers | 0.730 | 0.788 | 0.844 | 0.904 | 0.678 |
| **Dense on questions** | **0.852** | **0.880** | **0.910** | **0.944** | **0.799** |
| Answers + questions | 0.796 | 0.850 | 0.894 | 0.934 | 0.746 |
| Answers + BM25 | 0.662 | 0.736 | 0.826 | 0.902 | 0.605 |
| Answers + questions + BM25 | 0.742 | 0.816 | 0.900 | 0.934 | 0.667 |

### Question pipeline (`test_set_human.jsonl`, 500 queries)
- R@1 0.730, R@3 0.852, R@5 0.880, R@10 0.910.
- Context size: about 18 answers; mean 7,250 tokens, p90 9,732, max 16,483.
- "Gold answer kept" and "accepted answer kept" both scored 1.00, but can't fail, because the same logic picks both sides.

---

## File-by-file review

Status: **Open** means it's waiting for your decision or edit. **Fine** means no change is needed.

### Dataset and `01_split_qa.py`
| Finding | Evidence | Status |
|---|---|---|
| Parsing and the question/answer split are correct | 68,722 questions, 118,361 answers, 0 orphans, 0 missing accepted answers | Fine |
| Memory and speed | Peak 485 MB, about 2 seconds; with `close_reason`, 546 MB and about 5 seconds | Fine |
| Close reason isn't captured. `ClosedDate` is in `Posts.xml`; the reason is in `PostHistory.xml` | 10,720 closed questions; 8,265 are in the searchable corpus | Done 2026-11-05: `close_reason` added, tested on all 68,722 questions |
| Debug code on lines 52-59 can never use its first branch | `answers` is always a list | Done 2026-11-05: deleted |
| Scores are stored as text | Sorted as text, the top score is "99"; the real top is 1288. Later files convert before use | Fine for now |
| `Posts.xml` on disk differs from the archive | See 09-08 above | Fine: consider making `data/raw` read-only |

### `02_cleanup_qa.py`
Starting from 68,722 questions and 118,361 answers:

| Step | Questions left | Answers left | Status |
|---|---|---|---|
| Remove questions with no answers | 60,639 (8,083 removed) | 118,361 | Fine |
| New: remove off-topic and unclear (102, 2, 103) | 57,141 (3,498 removed) | 113,091 (5,270 removed) | Done 2026-11-05, confirmed by running the file |
| Remove near-duplicate questions, as built | 60,577 (62 removed, without the new filter) | 118,285 (76 removed) | Removed 2026-11-05 |
| Remove near-duplicate answers, as built | 60,577 | 118,284 (1 removed) | Removed 2026-11-05 |

| Finding | Evidence | Status |
|---|---|---|
| Duplicate detection wrongly removes different questions | In 14 sampled pairs, at least 3 are clearly unrelated, e.g. "How sqlmap works to get all database info" removed as a copy of "Simple solution to XSS" (0.912). Only 5 of the 62 are linked as duplicates on the site, and it caught only 4 of about 2,800 threads closed as duplicates. Removed questions had up to 31 votes | Removed 2026-11-05 |
| Stripping HTML doesn't fix it | With the filter on, 58 duplicates with HTML vs 44 without; 22 found only with HTML, 8 only without | Tested, not the fix |
| Comparing all questions at once needs about 29 GB (calculated, not run) | A batched version used under 2.1 GB and reproduced `threads.db` exactly in 214 seconds | Not needed if the step is removed |
| The "low thousands" size check is out of date | The result is about 57,000 questions | Removed 2026-11-05 |

### `03_parsing_answers.py` and `04_parsing_questions.py` (review not finished)
Tested on the filtered data by comparing the current `clean_body` with a version without `html.unescape`:

| Measure | Answers (113,091) | Questions (57,141) |
|---|---|---|
| Posts whose text changes with the fix | 2,932 (2.6%) | 2,406 (4.2%) |
| Posts currently losing text | 2,885 | 2,381 |
| Posts losing 50+ characters | 1,350 | 1,278 |
| Characters lost | 477,262 | 669,836 |
| Posts with more text before the fix (not investigated) | 46 | 25 |

| Finding | Evidence | Status |
|---|---|---|
| `html.unescape` deletes quoted code | The fix restores the text in answers 2179 and 2801 and question 10201 | Done in 03 on 2026-11-06 using shared `03_text_cleaning.py`, confirmed by running it. Open in 04 |
| The empty answer comes from the same bug | Answer 111466 is empty now, and not empty with the fix | Fixed by the same edit |
| `load_dotenv` can't simply be removed | `05_embed_qa.py` calls `OpenAI()` with no `load_dotenv`, so it relies on 03 and 04 for the API key | Open: add `load_dotenv` to 05 when removing it from 03 and 04 |
| Unused `client`, `MAX_TOKENS`, `encoding`, `tiktoken` | Other files only read `answer_records` from 03 | Open: remove |
| Extra spaces around inline tags, e.g. `AES -GCM`, `this paper .` | Reproduced. Changing this would change the text of most posts and force a full re-embed | Leave as is for now |

About 5,500 posts will need new embeddings: the 5,338 whose text changes, plus 58 questions and 73 answers that duplicate detection used to remove. Reuse all other embeddings.

---

## Decisions

### 2026-11-05: Filter closed threads by reason
Decision: remove off-topic and unclear threads from search. Mark them as not searchable instead of deleting them, so the choice can be reversed.

Close reasons in the searchable corpus, from the latest close event on questions still closed:

| Code | Meaning | Threads | Answers | Filter? |
|---|---|---|---|---|
| 101, 1 | Duplicate | 2,810 | 5,079 | Keep |
| 102, 2 | Off-topic | 2,551 | 3,882 | **Remove** |
| 103 | Unclear | 943 | 1,384 | **Remove** |
| 4 | Not a real question | 115 | 176 | Undecided: closest old code to "unclear" |
| 104 | Too broad | 1,135 | 1,914 | Keep |
| 105 | Opinion-based | 572 | 1,062 | Keep |
| 3 | Subjective | 93 | 174 | Keep |
| 7 | Too localized | 46 | 95 | Keep |

- Removing 102, 2 and 103 takes out 3,494 threads (5.8%) and 5,266 answers (4.5%).
- Off-topic and unclear threads were 62 of the 1,500 top-3 results in answer search (4.1%), so expect little change in scores. Measure it.
- 378 questions were closed and later reopened, so don't filter on history alone.
- Some removed threads have high votes, e.g. "Difference between XSS and CSRF?" (39), "Are there 'secure' languages?" (57).
- 3 of the 100 threads in each test set are off-topic (102). Their 15 queries in `test_set_human.jsonl` need removing or they become automatic misses.
- For duplicates, the original question's ID is in the `Text` field of the close event (`OriginalQuestionIds`).

### 2026-11-05: Move retrieval towards production
Decision: replace the local files (Chroma, SQLite, BM25 pickle, in-memory lists) with a real database and a multi-user API, up to the retrieval step. Proposed database: Postgres with pgvector. Details to be recorded here as each file is converted.

---

## Open questions
- Why was the duplicate-title test deleted on 10-16?
- How does question search score on that duplicate-title test?
- Should code 4 (not a real question) also be filtered?
- Should high-vote threads be exempt from the filter?
