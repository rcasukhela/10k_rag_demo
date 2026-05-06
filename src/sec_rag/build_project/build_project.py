from pathlib import Path

import yaml

from sentence_transformers import CrossEncoder
from huggingface_hub import snapshot_download

from sec_rag.build_project.project_structure import (
    DIRS,
    MODELS_DIR
)


def build():
    for path in DIRS:
        path.mkdir(parents=True, exist_ok=True)

    # # Save cross-encoder model
    # def load_hf_token(path=Path.cwd()/'secrets'/'secrets.yml'):
    #     with open(path, 'r') as f:
    #         return yaml.safe_load(f)['HF_token']
    # token = load_hf_token()
    token = None
    snapshot_download(
        repo_id="cross-encoder/ms-marco-MiniLM-L6-v2",
        local_dir=str(MODELS_DIR / 'ms-marco-MiniLM-L6-v2'),
        token=token,
    )


if __name__ == "__main__":
    build()
    print("Project directories ready.")