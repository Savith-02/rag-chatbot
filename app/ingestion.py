# app/ingestion.py

import io
import logging
import threading
from pathlib import Path
from typing import List, Dict, Set
from datetime import datetime

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .vectorstore import get_vectorstore, insert_documents
from .config import RAW_FILES_PATH, PROCESSED_FILES_TRACKER

logger = logging.getLogger(__name__)

# Lock to prevent concurrent folder ingestion
_ingestion_lock = threading.Lock()

def split_into_chunks(text: str, chunk_size: int = 600, chunk_overlap: int = 100) -> List[str]:
    """
    Split text into chunks using LangChain's RecursiveCharacterTextSplitter.
    This splitter respects semantic boundaries (paragraphs, sentences, words)
    and includes overlap between chunks for better context preservation.
    
    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    text = text.strip()
    if not text:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_text(text)
    return chunks


def ingest_pdf_bytes(file_bytes: bytes, file_name: str) -> dict:
    """
    Ingest a single PDF file (as bytes) into Milvus using langchain-milvus.
    """

    logger.info(f"-- Ingesting PDF: {file_name} --")
    reader = PdfReader(io.BytesIO(file_bytes))
    logger.info(f"page count: {len(reader.pages)}")
    docs: List[Document] = []
    pages_with_no_text = 0
    for page_index, page in enumerate(reader.pages):
        page_num = page_index + 1
        page_text = page.extract_text() or ""
        if not page_text.strip():
            pages_with_no_text += 1
            continue

        chunks = split_into_chunks(page_text, chunk_size=1200, chunk_overlap=150)

        for chunk_idx, chunk_text in enumerate(chunks):
            # Prepend filename to chunk content
            chunk_with_filename = f"[File: {file_name}]\n{chunk_text}"
            
            docs.append(
                Document(
                    page_content=chunk_with_filename,
                    metadata={
                        "chunk_id": f"{file_name}_{page_num}_{chunk_idx}",
                        "source": file_name,      
                        "file_name": file_name,
                        "page_start": page_num,
                        "page_end": page_num,
                        "chunk_index": chunk_idx,
                        "section_title": None,      # fill when you add section logic
                        "section_type": "text",     # "text" / "table" / etc
                    },
                )
            )

    if not docs:
        logger.info(f"-- No text extracted from {file_name} --\n\n")
        return {"file_name": file_name, "chunks_indexed": 0}

    # Insert documents with dense vectors
    num_inserted = insert_documents(docs)

    logger.info(f"-- Indexed {num_inserted} chunks from {file_name} with dense vectors --\n\n")
    return {"file_name": file_name, "chunks_indexed": num_inserted}


def load_processed_files() -> Set[str]:
    """
    Load the set of already-processed file names from the tracker file.
    """
    tracker_path = Path(PROCESSED_FILES_TRACKER)
    if not tracker_path.exists():
        return set()
    
    with open(tracker_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def mark_file_as_processed(file_name: str) -> None:
    """
    Append a file name to the processed files tracker.
    """
    tracker_path = Path(PROCESSED_FILES_TRACKER)
    with open(tracker_path, "a", encoding="utf-8") as f:
        f.write(f"{file_name}\n")


def ingest_folder() -> Dict:
    """
    Scan the RAW_FILES_PATH folder for PDF files.
    Process only those that haven't been processed before.
    Track processed files to avoid re-processing.
    
    Thread-safe: Uses a lock to prevent concurrent executions.
    
    Returns a summary of the ingestion operation.
    """
    # Try to acquire lock, skip if already running
    if not _ingestion_lock.acquire(blocking=False):
        logger.warning("Ingestion already in progress, skipping this run")
        return {
            "status": "skipped",
            "message": "Previous ingestion still running",
        }
    
    try:
        return _ingest_folder_impl()
    finally:
        _ingestion_lock.release()


def _ingest_folder_impl() -> Dict:
    """
    Internal implementation of folder ingestion.
    Should only be called by ingest_folder() which handles locking.
    """
    raw_files_dir = Path(RAW_FILES_PATH)
    if not raw_files_dir.exists():
        return {
            "status": "error",
            "message": f"Raw files directory does not exist: {RAW_FILES_PATH}",
        }
    
    # Load already processed files
    processed_files = load_processed_files()
    logger.info(f"Loaded {len(processed_files)} processed files")
    
    # Find all PDF files in the directory
    pdf_files = list(raw_files_dir.glob("*.pdf"))
    
    results = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "files_processed": [],
        "files_skipped": [],
        "files_failed": [],
    }
    
    for pdf_path in pdf_files:
        file_name = pdf_path.name
        
        # Skip if already processed
        if file_name in processed_files:
            results["skipped"] += 1
            results["files_skipped"].append(file_name)
            logger.warning(f"Skipping already processed file: {file_name}")
            continue
        
        try:
            # Read and process the file
            with open(pdf_path, "rb") as f:
                file_bytes = f.read()
            
            # Ingest the PDF
            ingest_result = ingest_pdf_bytes(file_bytes=file_bytes, file_name=file_name)
            
            # Mark as processed
            mark_file_as_processed(file_name)
            
            results["processed"] += 1
            results["files_processed"].append({
                "file_name": file_name,
                "chunks_indexed": ingest_result["chunks_indexed"],
            })
            logger.info(f"Processed file: {file_name}")

        except Exception as e:
            results["failed"] += 1
            results["files_failed"].append({
                "file_name": file_name,
                "error": str(e),
            })
            logger.error(f"Failed to process file: {file_name} - Error: {str(e)}", exc_info=True)
    
    return results
