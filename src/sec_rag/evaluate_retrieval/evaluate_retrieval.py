import os
import json
# import requests

from sec_rag.retrieval import (
    load_reranker,
    retrieve
)

if __name__ == '__main__':
    queries_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queries.jsonl')
    
    with open(queries_file, 'r') as f:
        results = [json.loads(line) for line in f]
    reranker = load_reranker()
    for result in results:
        query = result['query']
        retrieved_chunks = retrieve(query, reranker)

        print(query)

        total = len(retrieved_chunks)
        
        # any-term precision@k
        hits = 0
        for rc in retrieved_chunks:
            if any(item.lower() in rc.text.lower() for item in result['required_terms']):
                hits += 1
        print(f"any-term precision@{total}: {hits}/{total} = {hits / total:.2%}")

        # all-term precision@k
        hits = 0
        for rc in retrieved_chunks:
            if all(item.lower() in rc.text.lower() for item in result['required_terms']):
                hits += 1
        print(f"all-term precision@{total}: {hits}/{total} = {hits / total:.2%}")

        # top1 any-term hit
        top_chunk = max(retrieved_chunks, key=lambda c: c.reranker_score)
        hit = False
        if any(top_chunk.text.lower() for item in result['required_terms']):
            hit += True
        print(f'top1 any-term hit: {hit}')