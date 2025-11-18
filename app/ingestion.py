# app/ingestion.py

import io
import logging
from pathlib import Path
from typing import List, Dict, Set
from datetime import datetime

from pypdf import PdfReader
from langchain_core.documents import Document

from .vectorstore import get_vectorstore, insert_documents
from .config import RAW_FILES_PATH, PROCESSED_FILES_TRACKER

logger = logging.getLogger(__name__)

def split_into_chunks(text: str, max_chars: int = 1000) -> List[str]:
    """
    Very simple char-based splitter.
    Replace with a smarter section-aware splitter later.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start = end
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

        logger.info(f"Processing page {page_num}")

        chunks = split_into_chunks(page_text, max_chars=1000)

        for chunk_idx, chunk_text in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "chunk_id": f"{file_name}_{page_num}_{chunk_idx}",
                        "source": file_name,        # Required by Milvus schema
                        "file_name": file_name,
                        "page_start": page_num,
                        "page_end": page_num,
                        "chunk_index": chunk_idx,
                        "section_title": None,      # fill when you add section logic
                        "section_type": "text",     # "text" / "table" / etc
                    },
                )
            )

    logger.info(f"{pages_with_no_text} pages had no text extracted")
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
    
    Returns a summary of the ingestion operation.
    """
    raw_files_dir = Path(RAW_FILES_PATH)
    if not raw_files_dir.exists():
        return {
            "status": "error",
            "message": f"Raw files directory does not exist: {RAW_FILES_PATH}",
            "processed": 0,
            "skipped": 0,
            "failed": 0,
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
