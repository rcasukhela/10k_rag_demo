# %%

import json
import yaml
import pickle
from pathlib import Path

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
# with open(BM25_RETRIEVAL_PARAMS / 'retrieval_policy_v1.yml', 'r') as f:
#     bm25_retrieval_params = yaml.load(f, Loader=yaml.FullLoader)
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
    for rank, (i, score) in enumerate(top, start=1):
        chunk = chunks[i]
        result = {
            "rank": rank,
            "list_index": i,
            "score": float(score),
            "chunk_id": chunk.get("chunk_id"),
            "source": chunk.get("source"),
            "text": chunk["text"],
        }
        results.append(result)
    
    return results





# %%
with open()
reranker = CrossEncoder(str(MODELS_DIR))

query = 'in 2022, what was the main risk to the firm?'

bm25_chunks = get_top_k_chunks(query, chunks, top_k, bm25, verbose=False)





# %%