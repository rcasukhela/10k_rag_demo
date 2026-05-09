from pathlib import Path
import json
import yaml

import requests
import fitz

from sec_rag.build_project.project_structure import (
    RAW_DIR,
    PROCESSED_DIR,
    SEC_10KS_PATH,
    CHUNK_POLICY_PARAMS
)

from sec_rag.schemas import (
    Chunk
)

def write_chunks():
    # Reading YAML data from file
    with open(SEC_10KS_PATH / 'jpm_10ks.yml', 'r') as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)

    pdf_urls = {}
    for k,v in yaml_data['JPMC'].items():
        pdf_urls[f'jpm_{k}_10k.pdf'] = v


    for file, url in pdf_urls.items():
        out_path = RAW_DIR / file

        if out_path.exists():
            print(f'{out_path} already exists, skipping.')
            continue

        response = requests.get(url, timeout=5)
        
        out_path.write_bytes(response.content)
        print(f'saved: {out_path}')
    # Chunking
    '''Chunking policy: stick with a chunk size of 1500 characters
    and an overlap of 500 characters.
    That's a chunk size of around 300 words and
    an overlap of around 50 words, which is a good place to start.
    '''
    # Reading YAML data from file
    with open(CHUNK_POLICY_PARAMS / 'chunk_policy_v1.yml', 'r') as f:
        chunk_policy = yaml.load(f, Loader=yaml.FullLoader)
    chunk_size = chunk_policy['chunk_size']
    overlap = chunk_policy['overlap']
    def extract_pdf_text(path):
        with fitz.open(path) as pdf:
            return '\n'.join(page.get_text() or '' for page in pdf) # type: ignore
    PROCESSED_PATH = PROCESSED_DIR / 'chunks.jsonl'

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.write_text("", encoding="utf-8")

    i = 0
    for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
        with PROCESSED_PATH.open("a", encoding="utf-8") as f:
            print('processing file:', pdf_path)
            doc_id = pdf_path.stem
            year = int(doc_id.split("_")[1])

            text = extract_pdf_text(pdf_path)

            start = 0
            j = 0
            while start < len(text):
                if i % 100 == 0:
                    print('on chunk:', i)
                chunk = f'year: {year}, text: {text[start : start+chunk_size]}'
                start += chunk_size - overlap
                
                row = Chunk(
                    chunk_id = f"{doc_id}_{j:04d}",
                    doc_id = doc_id,
                    year = year,
                    filename = pdf_path.name,
                    doc_chunk_index = j,
                    chunk_index = i,
                    text = chunk,
                )

                f.write(row.model_dump_json() + "\n")
                
                i += 1
                j += 1

    return None