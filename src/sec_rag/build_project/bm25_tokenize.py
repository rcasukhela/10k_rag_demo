import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi
from sec_rag.build_project.spacy_regex_tokenize import spacy_regex_tokenize
from sec_rag.load_artifacts import load_chunks

from sec_rag.build_project.project_structure import (
    CHUNKS_PATH,
    
    BM25_PATH,
    BM25_META_PATH
)

def bm25_tokenize():
    chunks = load_chunks()
    docs = [spacy_regex_tokenize(chunk.text) for chunk in chunks]
    bm25 = BM25Okapi(docs)
    INDEX_META = {
        "chunks_path": str(CHUNKS_PATH),
        "tokenizer": "spacy_and_regex_v1",
        "bm25": "BM25Okapi_default",
    }
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    with open(BM25_META_PATH, 'w', encoding='utf-8') as f:
        json.dump(INDEX_META, f, indent=2)
    
    return None