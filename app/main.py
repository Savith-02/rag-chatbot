# app/main.py

from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from .ingestion import ingest_pdf_bytes
from .retrieval import query_docs

app = FastAPI(title="Financial RAG API with Milvus")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and index it into Milvus via langchain-milvus.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    result = ingest_pdf_bytes(file_bytes=file_bytes, file_name=file.filename)
    return result


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    file_name: Optional[str] = None


@app.post("/query")
async def query(req: QueryRequest):
    """
    Query the indexed chunks and get results with citations.
    """
    results = query_docs(
        query=req.query,
        top_k=req.top_k,
        file_name_filter=req.file_name,
    )
    return {"results": results}
