# app/ingestion.py

import io
import uuid
from typing import List

from pypdf import PdfReader
from langchain_core.documents import Document

from .vectorstore import get_vectorstore


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
    reader = PdfReader(io.BytesIO(file_bytes))
    docs: List[Document] = []

    for page_index, page in enumerate(reader.pages):
        page_num = page_index + 1
        page_text = page.extract_text() or ""
        if not page_text.strip():
            continue

        chunks = split_into_chunks(page_text, max_chars=1000)

        for chunk_idx, chunk_text in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "chunk_id": str(uuid.uuid4()),
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
        return {"file_name": file_name, "chunks_indexed": 0}

    vectorstore = get_vectorstore()
    vectorstore.add_documents(docs)

    return {"file_name": file_name, "chunks_indexed": len(docs)}
