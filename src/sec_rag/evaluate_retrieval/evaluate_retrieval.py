import os
import json
# import requests

from sec_rag.retrieval import (
    retrieve
)

if __name__ == '__main__':
    queries_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queries.jsonl')
    
    with open(queries_file, 'r') as f:
        results = [json.loads(line) for line in f]
    
    for result in results:
        query = result['query']
        retrieved_chunks = retrieve(query)

        print(query)

        total = len(retrieved_chunks)
        hits = 0
        for rc in retrieved_chunks:
            if any(item.lower() in rc.text.lower() for item in result['required_terms']):
                hits += 1
        print(f"any-term precision@{total}: {hits}/{total} = {hits / total:.2%}")