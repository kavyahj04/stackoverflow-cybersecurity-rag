from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
model = CrossEncoder(MODEL_NAME)


def rerank(question, candidates):
    pairs = [(question, hit["text"]) for hit in candidates]
    scores = model.predict(pairs)
    reranked = sorted(zip(candidates, scores), key = lambda x: x[1], reverse=True)
    return [{**hit, "score":float(score)} for hit, score in reranked]