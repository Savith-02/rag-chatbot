# app/vectorstore.py

import logging
from typing import Optional, List, Dict, Any
from pymilvus import MilvusClient, DataType
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

# Global MilvusClient instance
_client: Optional[MilvusClient] = None


def get_milvus_client() -> MilvusClient:
    """Get or create the global MilvusClient instance."""
    global _client
    if _client is None:
        _client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")
    return _client




def insert_documents(docs: list) -> int:
    """
    Insert documents with dense vectors.
    
    Args:
        docs: List of LangChain Document objects
    
    Returns:
        Number of documents inserted
    """
    if not docs:
        logger.warning("No documents to insert")
        return 0
    
    logger.info(f"Starting hybrid insertion for {len(docs)} documents")
    
    try:
        # Get client
        client = get_milvus_client()
        
        # Prepare data for insertion
        texts = [doc.page_content for doc in docs]
        logger.debug(f"Extracted {len(texts)} text chunks")
        
        # Generate dense embeddings
        logger.debug("Generating dense embeddings...")
        dense_vectors = embeddings.embed_documents(texts)
        logger.debug(f"Generated {len(dense_vectors)} dense vectors")
    except Exception as e:
        logger.error(f"Error during embedding generation: {str(e)}", exc_info=True)
        raise
    
    # Prepare data for insertion
    data = []
    for i, doc in enumerate(docs):
        md = doc.metadata or {}
        record = {
            "vector": dense_vectors[i],
            "text": doc.page_content,
            "source": md.get("source", ""),
            "file_name": md.get("file_name", ""),
            "chunk_id": md.get("chunk_id", ""),
            "page_start": int(md.get("page_start", 0)),
            "page_end": int(md.get("page_end", 0)),
            "chunk_index": int(md.get("chunk_index", 0)),
            "section_type": md.get("section_type", "text"),
        }
        
        # Add optional section_title if present
        if md.get("section_title"):
            record["section_title"] = md["section_title"]
        
        data.append(record)
    
    logger.debug(f"Prepared {len(data)} records for insertion")
    
    try:
        # Insert using new MilvusClient API
        logger.debug("Inserting data into Milvus collection...")
        insert_result = client.insert(
            collection_name=COLLECTION_NAME,
            data=data
        )
        logger.debug(f"Insert result: {insert_result}")
        
        logger.info(f"Successfully inserted {len(data)} documents with dense vectors")
        return len(data)
    except Exception as e:
        logger.error(f"Error during Milvus insertion: {str(e)}", exc_info=True)
        logger.error(f"Sample record structure: {data[0] if data else 'No data'}")
        raise


def create_collection_schema():
    """
    Define the schema for Milvus collection using new MilvusClient API.
    Dense vectors only.
    """
    # Create schema using new API
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    
    # Add primary key
    schema.add_field(
        field_name="pk",
        datatype=DataType.VARCHAR,
        is_primary=True,
        auto_id=True,
        max_length=100,
    )
    
    # Add text field
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=65535,
    )
    
    # Add dense vector field
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=EMBEDDING_DIM,
    )
    
    # Add metadata fields
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="page_start", datatype=DataType.INT64)
    schema.add_field(field_name="page_end", datatype=DataType.INT64)
    schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
    schema.add_field(field_name="section_type", datatype=DataType.VARCHAR, max_length=50)
    
    return schema


def ensure_collection_exists() -> None:
    """
    Ensure the collection exists with proper schema using new MilvusClient API.
    Dense vector search only.
    """
    client = get_milvus_client()
    
    # Check if collection exists
    if client.has_collection(COLLECTION_NAME):
        logger.info(f"Collection '{COLLECTION_NAME}' already exists.")
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return
    
    # Create schema
    schema = create_collection_schema()
    
    # Prepare index parameters
    index_params = MilvusClient.prepare_index_params()
    
    # Add index for dense vector
    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 1024},
    )
    
    # Create collection with schema and indexes
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    
    logger.info(f"Collection '{COLLECTION_NAME}' created with dense vector search")
    print(f"Collection '{COLLECTION_NAME}' created successfully.")


def get_vectorstore() -> Milvus:
    """
    Creates (or returns) a Milvus-backed LangChain VectorStore.
    Note: Collection must be initialized at startup via ensure_collection_exists().
    Note: LangChain integration still uses connection_args format.
    """
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={
            "uri": f"http://{MILVUS_HOST}:{MILVUS_PORT}",
        },
        # Search parameters for dense vector search
        search_params={
            "metric_type": "IP",
            "params": {"nprobe": 10},
        },
        # Specify which fields to return
        text_field="text",
        vector_field="vector",
        primary_field="pk",
        auto_id=True,
    )
    
    return vectorstore
