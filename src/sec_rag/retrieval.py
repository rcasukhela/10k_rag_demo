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

from sec_rag.schemas import (
    BM25Result,
    RerankedResult,
)

from sec_rag.build_project.spacy_regex_tokenize import (
    spacy_regex_tokenize
)

def load_reranker():
    # Load policies.
    with open(CONFIG_DIR / 'config_versions.yml', 'r') as f:
        policy_versions = yaml.safe_load(f)

    reranker_policy_version = policy_versions['reranker_policy_version']
    reranker_policy = load_policy('reranker_policy', reranker_policy_version)
    
    return CrossEncoder(str(MODELS_DIR / reranker_policy['reranker_model']))

def get_top_k_chunks(query, chunks, top_k, bm25, verbose=False):
    bm25_scores = bm25.get_scores(spacy_regex_tokenize(query))

    top = sorted(
        enumerate(bm25_scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    if verbose:
        for i, score in top:
            chunk = chunks[i]

            print("\n---")
            print("bm25 score:", round(float(score), 2))
            print("chunk_id:", chunk.get("chunk_id"))
            print("filename:", chunk.get("filename"))
            print(chunk.text[:100])
            print('\n-----')

    results = []
    for rank, bm25_retrieval_tuple in enumerate(top, start=1):
        chunk = chunks[bm25_retrieval_tuple[0]]
        list_index = bm25_retrieval_tuple[0]
        score = bm25_retrieval_tuple[1]
        result = BM25Result(
            **chunk.model_dump(),
            bm25_rank = rank,
            list_index = int(list_index),
            bm25_score = float(score)
        )
        results.append(result)
    
    return results

def retrieve(query, reranker):
    # Load policies.
    with open(CONFIG_DIR / 'config_versions.yml', 'r') as f:
        policy_versions = yaml.safe_load(f)
    
    bm25_retrieval_policy_version = policy_versions['bm25_retrieval_policy_version']
    bm25_retrieval_params = load_policy('bm25_retrieval_policy', bm25_retrieval_policy_version)
    top_k = bm25_retrieval_params.get('top_bm25_k')

    reranker_policy_version = policy_versions['reranker_policy_version']
    reranker_policy = load_policy('reranker_policy', reranker_policy_version)
    top_rerank_k = reranker_policy['top_rerank_k']

    # Load artifacts.
    chunks = load_chunks()
    bm25 = load_bm25()

    # BM25 retrieval.
    bm25_chunks = get_top_k_chunks(query, chunks, top_k, bm25, verbose=False)

    # Reranking.
    scores = reranker.predict([
        (query, chunk.text)
        for chunk in bm25_chunks
        ])

    ranked_idx = np.argsort(scores)[::-1][:top_rerank_k]

    reranked_chunks = []
    for final_rank, i in enumerate(ranked_idx, start=1):
        bm25_chunk = bm25_chunks[i].copy()
        reranked_chunk = RerankedResult(
            **bm25_chunk.model_dump(),
            final_rank = final_rank,
            reranker_score = float(scores[i])
        )
        reranked_chunks.append(reranked_chunk)

    return reranked_chunks





if __name__ == '__main__':
    reranker = load_reranker()
    reranked_chunks = retrieve("what is JPM's main potential for growth?", reranker)
    for reranked_chunk in reranked_chunks:
        print('chunk_id:', reranked_chunk.chunk_id)
        print('bm25 score:', reranked_chunk.bm25_score)
        print('reranker score:', reranked_chunk.reranker_score)
        print('text:', reranked_chunk.text[:100])
        print('\n')