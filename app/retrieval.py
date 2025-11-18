# app/retrieval.py

import logging
from typing import List, Dict, Any

from langchain_core.documents import Document

from .vectorstore import (
    get_vectorstore,
    embeddings,
)

logger = logging.getLogger(__name__)


def query_docs(
    query: str,
    top_k: int = 5,
    file_name_filter: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Query documents using dense vector search.
    
    Args:
        query: Search query
        top_k: Number of results to return
        file_name_filter: Optional filter by file name
    
    Returns:
        List of results with chunks and citations
    """
    logger.info(f"Dense vector search: '{query[:50]}...' (top_k={top_k})")
    
    # Dense vector search using LangChain wrapper
    vectorstore = get_vectorstore()
    
    kwargs = {}
    if file_name_filter:
        kwargs["filter"] = f'file_name == "{file_name_filter}"'
    
    docs_and_scores = vectorstore.similarity_search_with_score(
        query, k=top_k, **kwargs
    )
    
    # Format results
    results: List[Dict[str, Any]] = []
    for doc, score in docs_and_scores:
        md = doc.metadata or {}
        results.append(
            {
                "chunk_id": md.get("chunk_id"),
                "file_name": md.get("file_name"),
                "section_title": md.get("section_title"),
                "section_type": md.get("section_type"),
                "page_start": md.get("page_start"),
                "page_end": md.get("page_end"),
                "chunk_index": md.get("chunk_index"),
                "content": doc.page_content,
                "score": float(score),
            }
        )
    
    logger.info(f"Dense search returned {len(results)} results")
    return results
