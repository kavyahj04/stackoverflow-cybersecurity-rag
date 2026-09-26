from collections import defaultdict

def reciprocal_rank_fusion(dense_hits, sparse_hits, k=60, top_n=40):
    rrf_scores = defaultdict(float)
    info_by_id = {}

    for hits in (dense_hits, sparse_hits):
        for rank, hit in enumerate(hits, start=1):
            answer_id = hit["answer_id"]
            rrf_scores[answer_id] += 1/(k+rank)
            info_by_id[answer_id] = {"text": hit["text"], "url": hit["url"]}
        
    ranked_ids = sorted(rrf_scores, key= lambda aid: rrf_scores[aid], reverse=True)[:top_n]

    return[
        {
            "answer_id":answer_id,
            "score":rrf_scores[answer_id],
            "text": info_by_id[answer_id]["text"],
            "url": info_by_id[answer_id]["url"]
        }
        for answer_id in ranked_ids
    ]