import json

def load_chunks(CHUNKS_PATH):
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        chunks = [json.loads(line) for line in f]
    return chunks