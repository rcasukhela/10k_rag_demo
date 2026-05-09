from pathlib import Path

import yaml

from sentence_transformers import CrossEncoder
from huggingface_hub import snapshot_download

from sec_rag.build_project.project_structure import (
    DIRS,
    MODELS_DIR
)

from sec_rag.build_project.write_chunks import (
    write_chunks
)

from sec_rag.build_project.bm25_tokenize import (
    bm25_tokenize
)


def build():
    print('Initializing project directories...')
    for path in DIRS:
        path.mkdir(parents=True, exist_ok=True)
    print("Project directories ready.")

    print('Downloading cross-encoder reranker from HF Hub...')
    # Save cross-encoder model
    def load_hf_token(path=Path.cwd()/'secrets'/'secrets.yml') -> str | None:
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)['HF_token']
        except:
            return None
    token = load_hf_token()
    snapshot_download(
        repo_id="cross-encoder/ms-marco-MiniLM-L6-v2",
        local_dir=str(MODELS_DIR / 'ms-marco-MiniLM-L6-v2'),
        token=token,
    )
    print('Reranker downloaded.')

    print('Downloading corpus and writing chunks...')
    write_chunks()
    print('Complete.')

    print('Tokenizing and creating BM25 index...')
    bm25_tokenize()
    print('Complete.')


if __name__ == "__main__":
    build()
    