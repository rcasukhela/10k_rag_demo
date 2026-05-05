from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PICKLE_DIR = ARTIFACTS_DIR / "pickle"
META_DIR = ARTIFACTS_DIR / "meta"

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
BM25_PATH = PICKLE_DIR / "bm25.pkl"
BM25_META_PATH = META_DIR / "bm25_meta.json"

# Build-specific dirs
BUILD_DIR = PROJECT_ROOT / 'src' / 'sec_rag'
SEC_10KS_PATH = BUILD_DIR / 'config' / 'sec_10ks'
CHUNK_POLICY_PARAMS = BUILD_DIR / 'config' / 'chunk_policy'
BM25_RETRIEVAL_PARAMS = BUILD_DIR / 'config' / 'bm25_retrieval_policy'

DIRS = [
    RAW_DIR,
    PROCESSED_DIR,
    ARTIFACTS_DIR,
    PICKLE_DIR,
    META_DIR,
]