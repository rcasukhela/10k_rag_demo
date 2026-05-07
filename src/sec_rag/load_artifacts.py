import pickle
import json

from sec_rag.build_project.project_structure import (
    CHUNKS_PATH,
    BM25_PATH
)

from sec_rag.schemas import (
    Chunk
)

def load_chunks():
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        chunks = [Chunk.model_validate_json(line) for line in f]
            
    return chunks

def load_bm25():
    with open(BM25_PATH, 'rb') as f:
        bm25 = pickle.load(f)
    return bm25