from pydantic import BaseModel, ConfigDict


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    doc_id: str
    year: int
    filename: str
    doc_chunk_index: int
    chunk_index: int
    text: str


class BM25Result(Chunk):
    bm25_rank: int
    list_index: int
    bm25_score: float


class RerankedResult(BM25Result):
    final_rank: int
    reranker_score: float