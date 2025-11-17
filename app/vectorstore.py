# app/vectorstore.py

from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

# You can parameterize these via env vars later
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "financial_docs"

# Instantiate embeddings once (local BGE-large)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")


def get_vectorstore() -> Milvus:
    """
    Creates (or returns) a Milvus-backed LangChain VectorStore.

    With langchain-milvus, the collection is auto-created if it doesn't exist.
    """
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={
            "host": MILVUS_HOST,
            "port": MILVUS_PORT,
        },
        # You can keep default search params for now; tune later.
        # search_params={"metric_type": "IP", "params": {"nprobe": 10}},
    )
    return vectorstore
