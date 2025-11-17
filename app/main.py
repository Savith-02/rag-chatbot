# app/main.py

import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler

from .ingestion import ingest_pdf_bytes, ingest_folder
from .retrieval import query_docs
from .config import RAW_FILES_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize scheduler
scheduler = BackgroundScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - start/stop scheduler"""
    # Startup: Start the scheduler
    logger.info("Starting background scheduler for folder ingestion...")
    scheduler.add_job(
        func=ingest_folder,
        trigger="interval",
        seconds=45,
        id="folder_ingestion_job",
        name="Ingest PDFs from raw_files folder",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler started. Folder ingestion will run every 45 seconds.")
    
    yield
    
    # Shutdown: Stop the scheduler
    logger.info("Shutting down scheduler...")
    scheduler.shutdown()
    logger.info("Scheduler stopped.")


app = FastAPI(
    title="Financial RAG API with Milvus",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and index it into Milvus via langchain-milvus.
    This directly processes the file without saving it to raw_files folder.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    result = ingest_pdf_bytes(file_bytes=file_bytes, file_name=file.filename)
    return result


@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF file to the raw_files folder.
    The file will be processed by the scheduled ingestion task.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    logger.info(f"\n-- Uploading file: {file.filename} --")
    # Save file to raw_files directory
    raw_files_dir = Path(RAW_FILES_PATH)
    raw_files_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = raw_files_dir / file.filename
    
    # Check if file already exists
    if file_path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"File '{file.filename}' already exists in raw_files folder"
        )
    
    # Write file to disk
    file_bytes = await file.read()
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    logger.info(f"-- File uploaded successfully: {file.filename} --")
    return {
        "status": "success",
        "message": f"File '{file.filename}' uploaded successfully",
        "file_path": str(file_path),
        "note": "File will be processed by the scheduled ingestion task"
    }


@app.post("/trigger_ingestion")
async def trigger_ingestion():
    """
    Manually trigger the folder ingestion process.
    This will process all unprocessed PDF files in the raw_files folder.
    """
    logger.info("Manual ingestion triggered via API")
    result = ingest_folder()
    return result


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    file_name: Optional[str] = None
    use_hybrid: bool = True  # Enable hybrid search by default


@app.post("/query")
async def query(req: QueryRequest):
    """
    Query the indexed chunks and get results with citations.
    Uses hybrid search (dense vector + BM25) by default for better financial document retrieval.
    
    Set use_hybrid=False to use dense-only vector search.
    """
    results = query_docs(
        query=req.query,
        top_k=req.top_k,
        file_name_filter=req.file_name,
        use_hybrid=req.use_hybrid,
    )
    return {"results": results}


@app.get("/ingestion_status")
async def ingestion_status():
    """
    Get the status of the scheduled ingestion job.
    """
    job = scheduler.get_job("folder_ingestion_job")
    if job:
        return {
            "status": "active",
            "job_id": job.id,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            "interval_seconds": 45,
        }
    return {"status": "inactive"}
