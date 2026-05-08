from fastapi import FastAPI
from sec_rag.retrieval import (
    load_reranker,
    retrieve
)

app = FastAPI()

reranker = load_reranker()

@app.get("/search")
def search(query):
    return {
        "query": query,
        "results": retrieve(query, reranker),
    }