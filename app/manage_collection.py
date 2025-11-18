#!/usr/bin/env python
"""
Utility script to manage Milvus collection.
Usage:
    python -m app.manage_collection --drop    # Drop existing collection
    python -m app.manage_collection --create  # Create collection with schema
    python -m app.manage_collection --info    # Show collection info
    python -m app.manage_collection --recreate   # Drop and recreate collection
"""

import argparse
from pymilvus import connections, utility, Collection
from .vectorstore import (
    MILVUS_HOST,
    MILVUS_PORT,
    COLLECTION_NAME,
    ensure_collection_exists,
)


def connect_to_milvus():
    """Connect to Milvus server"""
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
    )
    print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")


def drop_collection():
    """Drop the existing collection"""
    connect_to_milvus()
    
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' dropped successfully.")
    else:
        print(f"Collection '{COLLECTION_NAME}' does not exist.")


def create_collection():
    """Create collection with proper schema"""
    connect_to_milvus()
    ensure_collection_exists()

def recreate_collection():
    """Drop the existing collection and recreate"""
    connect_to_milvus()
    
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' dropped successfully.")
    else:
        print(f"Collection '{COLLECTION_NAME}' does not exist.")
    ensure_collection_exists()


def _format_field_params(field) -> str:
    """Format field parameters for display"""
    params = []
    if field.is_primary:
        params.append("primary")
    if hasattr(field, 'auto_id') and field.auto_id:
        params.append("auto_id")
    if hasattr(field, 'max_length') and field.max_length:
        params.append(f"max_length={field.max_length}")
    if hasattr(field, 'dim') and field.dim:
        params.append(f"dim={field.dim}")
    return ", ".join(params) if params else "-"


def show_collection_info():
    """Show information about the collection"""
    connect_to_milvus()
    
    if not utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' does not exist.")
        return
    
    collection = Collection(COLLECTION_NAME)
    collection.load()
    
    # Header
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"Collection: {COLLECTION_NAME}")
    print(separator)
    print(f"Description: {collection.description}")
    print(f"Number of entities: {collection.num_entities}")
    
    # Schema
    print(f"\nSchema:")
    print(f"{'  Field Name':<20} {'Type':<20} {'Params'}")
    print(f"  {'-'*58}")
    for field in collection.schema.fields:
        param_str = _format_field_params(field)
        print(f"  {field.name:<20} {str(field.dtype):<20} {param_str}")
    
    # Indexes
    print(f"\nIndexes:")
    for index in collection.indexes:
        print(f"  Field: {index.field_name}")
        print(f"  Index type: {index.params.get('index_type', 'N/A')}")
        print(f"  Metric type: {index.params.get('metric_type', 'N/A')}")
    
    print(f"{separator}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Manage Milvus collection for RAG chatbot"
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop the existing collection"
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create collection with schema"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show collection information"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate collection (WARNING: deletes all data)"
    )
    
    args = parser.parse_args()
    
    if args.recreate:
        drop_collection()
        create_collection()
    elif args.drop:
        drop_collection()
    elif args.create:
        create_collection()
    elif args.info:
        show_collection_info()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
