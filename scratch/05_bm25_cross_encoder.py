# %%

import json
import yaml
import pickle
from pathlib import Path

import numpy as np

from rank_bm25 import BM25Okapi

from sentence_transformers import CrossEncoder

from sec_rag.tokenize import tokenize
from sec_rag.load_chunks import load_chunks

from sec_rag.build_project.project_structure import (
    ARTIFACTS_DIR,
    CHUNKS_PATH,
    BM25_RETRIEVAL_PARAMS,
    MODELS_DIR,
)
from sec_rag.build_project.load_policy import (
    load_policy,
)





# %%
bm25_retrieval_params = load_policy('bm25_retrieval_policy', 'v1')
top_k = bm25_retrieval_params.get('top_k')





# %%
with open(ARTIFACTS_DIR / 'pickle' / 'bm25.pkl', 'rb') as f:
    bm25 = pickle.load(f)
    
with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
    chunks = [json.loads(line) for line in f]





# %%
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


# %%
reranker_model = load_policy('reranker_policy', 'v1')

# %%
reranker = CrossEncoder(str(MODELS_DIR / reranker_model['reranker_model']))

# %%
query = 'in 2022, what was the main risk to the firm?'

bm25_chunks = get_top_k_chunks(query, chunks, top_k, bm25, verbose=False)

# %%
scores = reranker.predict([(query, chunk['text']) for chunk in bm25_chunks])



# %%
reranked_chunks = [bm25_chunks[index] for index in np.argsort(scores)[::-1][:10]]