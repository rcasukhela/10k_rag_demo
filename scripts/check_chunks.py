import json
from pathlib import Path

# Should be run AFTER notebooks/01_process_raw_pdfs.ipynb is run.
CHUNKS_PATH = Path('data/processed/chunks.jsonl')

with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
    chunks = [json.loads(line) for line in f]

for chunk in chunks[:3]:
    print('chunk_id:', chunk.get('chunk_id'))
    print('source:', chunk.get('source'))
    print('text:', chunk.get('text')[:200])