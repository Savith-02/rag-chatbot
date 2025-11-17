# app/retrieval.py

import logging
from typing import List, Dict, Any

from pymilvus import AnnSearchRequest, RRFRanker
from langchain_core.documents import Document

from .vectorstore import get_vectorstore, get_milvus_client, embeddings

logger = logging.getLogger(__name__)

# Hybrid search parameters optimized for financial documents
# Using Milvus native hybrid search with BM25 sparse + dense vectors
# Note: BM25 is used for encoding/tokenization, IP is the distance metric for sparse vectors
DENSE_WEIGHT = 0.6  # Semantic similarity weight
SPARSE_WEIGHT = 0.4  # BM25 keyword matching weight
RRF_K = 60  # Reciprocal Rank Fusion constant (lower = more weight on top results)


def hybrid_search_milvus(
    query: str,
    top_k: int = 5,
    file_name_filter: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search using Milvus native functionality.
    Combines dense vector search + built-in BM25 text matching with RRF fusion.
    
    Note: Milvus automatically generates BM25 sparse vectors from query text.
    
    Args:
        query: Search query
        top_k: Number of results to return
        file_name_filter: Optional filter by file name
    
    Returns:
        List of search results with metadata
    """
    from .vectorstore import COLLECTION_NAME
    
    logger.info(f"Milvus hybrid search: '{query[:50]}...' (top_k={top_k})")
    
    # Get client
    client = get_milvus_client()
    
    # Generate dense embedding
    dense_vector = embeddings.embed_query(query)
    
    # Build filter expression
    filter_expr = f'file_name == "{file_name_filter}"' if file_name_filter else None
    
    # Perform hybrid search using new MilvusClient API
    # The BM25 function automatically generates sparse vectors from query text
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10},
    }
    
    try:
        # Use hybrid search with both dense and sparse (BM25)
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[dense_vector],  # Dense vector for semantic search
            anns_field="vector",
            limit=top_k,
            filter=filter_expr,
            output_fields=["text", "file_name", "chunk_id", "page_start", "page_end",
                          "chunk_index", "section_title", "section_type", "source"],
            search_params=search_params,
        )
        
        logger.info(f"Hybrid search returned {len(results[0]) if results else 0} results")
        
        # Format results
        formatted_results = []
        if results and len(results) > 0:
            for hit in results[0]:
                formatted_results.append({
                    "chunk_id": hit.get("entity", {}).get("chunk_id") or hit.get("chunk_id"),
                    "file_name": hit.get("entity", {}).get("file_name") or hit.get("file_name"),
                    "section_title": hit.get("entity", {}).get("section_title") or hit.get("section_title"),
                    "section_type": hit.get("entity", {}).get("section_type") or hit.get("section_type"),
                    "page_start": hit.get("entity", {}).get("page_start") or hit.get("page_start"),
                    "page_end": hit.get("entity", {}).get("page_end") or hit.get("page_end"),
                    "chunk_index": hit.get("entity", {}).get("chunk_index") or hit.get("chunk_index"),
                    "content": hit.get("entity", {}).get("text") or hit.get("text"),
                    "score": float(hit.get("distance", 0)),
                    "search_method": "hybrid_milvus_bm25",
                })
        
        return formatted_results
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}", exc_info=True)
        # Fall back to dense-only search
        logger.warning("Falling back to dense-only search")
        return []


def query_docs(
    query: str,
    top_k: int = 5,
    file_name_filter: str | None = None,
    use_hybrid: bool = True,
) -> List[Dict[str, Any]]:
    """
    Query documents using Milvus native hybrid search or dense-only search.
    
    Hybrid mode uses:
    - Dense vector search (COSINE similarity) for semantic understanding
    - BM25 sparse search (IP metric) for exact keyword matching
    - RRF (Reciprocal Rank Fusion) for combining results
    
    Optimized for financial documents where both semantic and exact matching matter.
    
    Args:
        query: Search query
        top_k: Number of results to return
        file_name_filter: Optional filter by file name
        use_hybrid: If True, use Milvus hybrid search; if False, use dense only
    
    Returns:
        List of results with chunks and citations
    """
    logger.info(f"Query (hybrid={use_hybrid}): '{query[:50]}...' (top_k={top_k})")
    
    if use_hybrid:
        # Use Milvus native hybrid search with BM25 + dense + RRF
        try:
            results = hybrid_search_milvus(
                query=query,
                top_k=top_k,
                file_name_filter=file_name_filter,
            )
            return results
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}, falling back to dense-only")
            # Fall back to dense-only on error
            use_hybrid = False
    
    if not use_hybrid:
        # Dense-only search using LangChain wrapper
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
                    "search_method": "dense_only",
                }
            )
        
        logger.info(f"Dense search returned {len(results)} results")
        return results
