import json
import yaml
import pickle
from pathlib import Path

import numpy as np

from rank_bm25 import BM25Okapi

from sentence_transformers import CrossEncoder

from sec_rag.build_project.project_structure import (
    CONFIG_DIR,
    MODELS_DIR
)

from sec_rag.build_project.load_policy import (
    load_policy
)

from sec_rag.load_artifacts import (
    load_chunks,
    load_bm25
)

from sec_rag.tokenize import (
    tokenize
)

def get_top_k_chunks(query, chunks, top_k, bm25, verbose=False):
    scores = bm25.get_scores(tokenize(query))

    top = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    if verbose:
        for i, score in top:
            chunk = chunks[i]

            print("\n---")
            print("score:", round(float(score), 2))
            print("chunk_id:", chunk.get("chunk_id"))
            print("source:", chunk.get("source"))
            print(chunk["text"][:100])
            print('\n-----')

    results = []
    for rank, bm25_retrieval_tuple in enumerate(top):
        chunk = chunks[bm25_retrieval_tuple[0]]
        list_index = bm25_retrieval_tuple[0]
        score = bm25_retrieval_tuple[1]
        result = {
            "rank": rank,
            "list_index": list_index,
            "score": float(score),
            "chunk_id": chunk.get("chunk_id"),
            "source": chunk.get("source"),
            "text": chunk["text"],
        }
        results.append(result)
    
    return results

def retrieval(query):
    with open(CONFIG_DIR / 'config_version.yml', 'r') as f:
        policy_versions = yaml.safe_load(f)
    
    bm25_retrieval_policy_version = policy_versions['bm25_retrieval_policy_version']
    bm25_retrieval_params = load_policy('bm25_retrieval_policy', bm25_retrieval_policy_version)
    top_k = bm25_retrieval_params.get('top_bm25_k')

    reranker_policy_version = policy_versions['reranker_policy_version']
    reranker_policy = load_policy('reranker_policy', reranker_policy_version)
    top_rerank_k = reranker_policy['top_rerank_k']

    chunks = load_chunks()
    bm25 = load_bm25()

    # BM25 retrieval.
    bm25_chunks = get_top_k_chunks(query, chunks, top_k, bm25, verbose=False)

    # Reranking.
    reranker = CrossEncoder(str(MODELS_DIR / reranker_policy['reranker_model']))
    scores = reranker.predict([
        (query, chunk['text'])
        for chunk in bm25_chunks
        ])

    reranked_chunks = [bm25_chunks[index] for index in np.argsort(scores)[::-1][:top_rerank_k]]

    return reranked_chunks





if __name__ == '__main__':
    print(retrieval('what is JPM''s main potential for growth?'))