# app/retrieval.py

from typing import List, Dict, Any

from .vectorstore import get_vectorstore


def query_docs(
    query: str,
    top_k: int = 5,
    file_name_filter: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Query documents in Milvus and return chunks + citations.
    """
    vectorstore = get_vectorstore()

    # Optional metadata filter example (by file_name)
    kwargs = {}
    if file_name_filter:
        # langchain-milvus uses Milvus term filter syntax internally.
        # This becomes a scalar filter expression in Milvus.
        kwargs["filter"] = f'file_name == "{file_name_filter}"'

    docs_and_scores = vectorstore.similarity_search_with_score(
        query, k=top_k, **kwargs
    )

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

    return results
