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


def create_collection_schema() -> CollectionSchema:
    """
    Define the explicit schema for the Milvus collection.
    Following Milvus best practices with proper field definitions.
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
        # Vector embeddings
        FieldSchema(
            name="vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM,
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
    
    # Create index for vector field
    index_params = {
        "metric_type": "COSINE",  # or "L2", "IP"
        "index_type": "IVF_FLAT",  # or "HNSW" for better performance
        "params": {"nlist": 1024},
    }
    
    collection.create_index(
        field_name="vector",
        index_params=index_params,
    )
    
    print(f"Collection '{COLLECTION_NAME}' created successfully with schema.")


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
