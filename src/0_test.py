import json
import pickle
from parsing_answers import answer_records

with open("bm25_answers.pkl", "rb") as file:
    data = pickle.load(file)

bm25 = data["bm25"]
ids = data["ids"]

# --- Pick the first answer in the index ---
first_id = ids[0]
record = next(r for r in answer_records if r["answer_id"] == first_id)

print("=" * 70)
print("1. ORIGINAL ANSWER RECORD (before BM25)")
print("   -> the full, human-readable chunk with text + metadata")
print("=" * 70)
print(json.dumps(record, indent=2))

print()
print("=" * 70)
print("2. HOW BM25 STORED THIS SAME ANSWER")
print("   -> only word counts survive, everything else is discarded")
print("=" * 70)
print(f"answer_id           : {first_id}")
print(f"doc_len (word count) : {bm25.doc_len[0]}")
print("doc_freqs (word -> count IN THIS ONE ANSWER ONLY):")
print(json.dumps(bm25.doc_freqs[0], indent=2))

print()
print("=" * 70)
print("3. WHERE RARITY (Count #2) IS STORED")
print("   -> NOT inside doc_freqs. It's ONE global dict shared by ALL")
print("      118,361 answers, not tied to answer_id 4 specifically.")
print("=" * 70)
sample_words = ["the", "i", "security", "certification", "ocsp"]
for word in sample_words:
    print(f"bm25.idf['{word}']  = {bm25.idf.get(word)}")

print()
print("=" * 70)
print("4. WHAT HAPPENS WHEN A NEW QUESTION COMES IN")
print("   -> query 'what is the ocsp certification' scored against answer #4")
print("=" * 70)
query_words = ["what", "is", "the", "ocsp", "certification"]
doc_len_0 = bm25.doc_len[0]
k1, b, avgdl = bm25.k1, bm25.b, bm25.avgdl

total_score = 0
for word in query_words:
    tf = bm25.doc_freqs[0].get(word, 0)          # Count #1: local count in THIS answer
    idf = bm25.idf.get(word, 0)                   # Count #2: global rarity
    saturated_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len_0 / avgdl)) if tf else 0
    contribution = idf * saturated_tf
    total_score += contribution
    print(f"  word={word:14s} local_count={tf}  rarity(idf)={idf:6.2f}  -> contributes {contribution:6.3f} to score")

print(f"\n  TOTAL SCORE for answer #4 on this query: {total_score:.3f}")
print("\n  Notice: 'what' and 'is' don't appear in this answer at all (count=0),")
print("  so they contribute NOTHING. 'the' appears but is common everywhere,")
print("  so its rarity is low -> small contribution. 'ocsp' and 'certification'")
print("  are rare across the corpus AND present in this answer -> they drive")
print("  almost the entire score. This is why rarity is needed: without it,")
print("  every matching word would count equally, and common filler words")
print("  would drown out the words that actually indicate relevance.")

print()
print("=" * 70)
print("5. THIS REPEATS FOR ALL 118,361 ANSWERS, THEN THEY ARE RANKED")
print("=" * 70)
scores = bm25.get_scores(query_words)
top5_idx = scores.argsort()[::-1][:5]
for rank, idx in enumerate(top5_idx, start=1):
    print(f"  #{rank}  answer_id={ids[idx]:8s} score={scores[idx]:.3f}")