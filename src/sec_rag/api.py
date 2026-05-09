from fastapi import FastAPI
from sec_rag.retrieval import (
    load_reranker,
    retrieve
)
from sec_rag.llm import (
    generate
)

app = FastAPI()

reranker = load_reranker()

def build_prompt(query, chunks):
    context = "\n\n".join(
        f"[Source {i+1} | {c.chunk_id} | {c.filename}]\n{c.text}"
        for i, c in enumerate(chunks[:3])
    )

    return f"""
Answer the question using only the context below.

If the context is not enough, say that the provided filings do not contain enough evidence.

Write 2-4 English sentences.
Do not quote the context as the answer.
Abstain from answering if context is deemed insufficient.
Do not list every sentence from the context.

Question:
{query}

Context:
{context}

Answer:
""".strip()

@app.get("/search")
def search(query):
    return {
        "query": query,
        "results": retrieve(query, reranker),
    }

@app.get('/ask')
def ask(query):
    chunks = retrieve(query, reranker)
    prompt = build_prompt(query, chunks)
    answer = generate(prompt)

    return {
        'query': query,
        'chunks': chunks[:3],
        'answer': answer
    }