# 10-K RAG Demo

# Setup
  * run `pip install -r requirements.txt`
  * run `python .\src\sec_rag\build_project\build_project.py`
  * run `docker compose up -d ollama`
  * run `docker compose exec ollama ollama pull <MODEL_NAME>` (I use gemma3:270m, change in `.env`)
  * run `docker compose up --build app`
  * test functionality with `http://localhost:8000/search?q=interest rates` in browser

# Method
  * Retrieval: BM25 (bag-of-words) + cross-encoder reranking
  * Generation: Ollama (local, open-source Transformers)