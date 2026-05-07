from fastapi import FastAPI, Query
from sec_rag.retrieval import retrieval

app = FastAPI()

@app.get("/search")
def search(query):
    return {
        "query": query,
        "results": retrieval(query),
    }