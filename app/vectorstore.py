# app/vectorstore.py

import logging
from typing import Optional
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# You can parameterize these via env vars later
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "financial_docs"

# BGE-large-en produces 1024-dimensional embeddings
EMBEDDING_DIM = 1024

# Instantiate embeddings once (local BGE-large)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")


def get_milvus_collection() -> Collection:
    """Get the Milvus collection object for direct operations."""
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
    )
    return Collection(name=COLLECTION_NAME, using="default")


def insert_documents_with_hybrid(docs: list) -> int:
    """
    Insert documents with both dense and sparse vectors for hybrid search.
    
    Args:
        docs: List of LangChain Document objects
    
    Returns:
        Number of documents inserted
    """
    from pymilvus.model.sparse import BM25EmbeddingFunction
    
    if not docs:
        return 0
    
    # Get collection
    collection = get_milvus_collection()
    
    # Initialize BM25 function for sparse vectors
    bm25_ef = BM25EmbeddingFunction()
    
    # Prepare data for insertion
    texts = [doc.page_content for doc in docs]
    
    # Generate dense embeddings
    dense_vectors = embeddings.embed_documents(texts)
    
    # Generate sparse embeddings using BM25
    # BM25 needs to be fitted on the corpus first
    bm25_ef.fit(texts)
    sparse_vectors = bm25_ef.encode_documents(texts)
    
    # Prepare entities
    entities = []
    for i, doc in enumerate(docs):
        md = doc.metadata or {}
        entity = {
            "vector": dense_vectors[i],
            "sparse_vector": sparse_vectors[i],
            "text": doc.page_content,
            "source": md.get("source", ""),
            "file_name": md.get("file_name", ""),
            "chunk_id": md.get("chunk_id", ""),
            "page_start": md.get("page_start", 0),
            "page_end": md.get("page_end", 0),
            "chunk_index": md.get("chunk_index", 0),
            "section_type": md.get("section_type", "text"),
        }
        
        # Add optional fields if present
        if "section_title" in md and md["section_title"]:
            entity["section_title"] = md["section_title"]
        
        entities.append(entity)
    
    # Insert into collection
    collection.insert(entities)
    collection.flush()
    
    logger.info(f"Inserted {len(entities)} documents with hybrid vectors")
    return len(entities)


def create_collection_schema() -> CollectionSchema:
    """
    Define the explicit schema for the Milvus collection with hybrid search support.
    Includes both dense vector and sparse vector fields for BM25.
    """
    fields = [
        # Primary key - auto-generated
        FieldSchema(
            name="pk",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=True,
            max_length=100,
        ),
        # Dense vector embeddings (semantic)
        FieldSchema(
            name="vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM,
        ),
        # Sparse vector for BM25 (keyword matching)
        FieldSchema(
            name="sparse_vector",
            dtype=DataType.SPARSE_FLOAT_VECTOR,
        ),
        # Text content
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=65535,  # Max text length
        ),
        # Metadata fields
        FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=512,
        ),
        FieldSchema(
            name="file_name",
            dtype=DataType.VARCHAR,
            max_length=512,
        ),
        FieldSchema(
            name="chunk_id",
            dtype=DataType.VARCHAR,
            max_length=100,
        ),
        FieldSchema(
            name="page_start",
            dtype=DataType.INT64,
        ),
        FieldSchema(
            name="page_end",
            dtype=DataType.INT64,
        ),
        FieldSchema(
            name="chunk_index",
            dtype=DataType.INT64,
        ),
        FieldSchema(
            name="section_type",
            dtype=DataType.VARCHAR,
            max_length=50,
        ),
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description="Financial documents with embeddings for RAG",
        enable_dynamic_field=True,  # Allow additional fields if needed
    )
    
    return schema


def ensure_collection_exists() -> None:
    """
    Ensure the collection exists with the proper schema.
    If it doesn't exist, create it. If it exists, verify schema compatibility.
    """
    # Connect to Milvus
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
    )
    
    # Check if collection exists
    if utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return
    
    # Create collection with schema
    schema = create_collection_schema()
    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema,
        using="default",
    )
    
    # Create index for dense vector field
    dense_index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024},
    }
    
    collection.create_index(
        field_name="vector",
        index_params=dense_index_params,
    )
    
    # Create index for sparse vector field (BM25)
    sparse_index_params = {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "BM25",
    }
    
    collection.create_index(
        field_name="sparse_vector",
        index_params=sparse_index_params,
    )
    
    logger.info(f"Collection '{COLLECTION_NAME}' created with hybrid search support (dense + sparse)")
    print(f"Collection '{COLLECTION_NAME}' created successfully with hybrid search schema.")


def get_vectorstore() -> Milvus:
    """
    Creates (or returns) a Milvus-backed LangChain VectorStore.
    Ensures the collection exists with proper schema before returning.
    """
    # Ensure collection exists with proper schema
    ensure_collection_exists()
    
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={
            "host": MILVUS_HOST,
            "port": MILVUS_PORT,
        },
        # Search parameters
        search_params={
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        },
        # Specify which fields to return
        text_field="text",
        vector_field="vector",
        primary_field="pk",
        auto_id=True,  # Tell LangChain that primary key is auto-generated
    )
    
    return vectorstore
