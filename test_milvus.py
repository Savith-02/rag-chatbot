from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Connect to Milvus standalone (running in Docker)
connections.connect(
    alias="default",
    host="127.0.0.1",
    port="19530",
)

print("Connected!")

collection_name = "quick_test_collection"

# Drop if exists for a clean test
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4),
]

schema = CollectionSchema(fields, description="Quick test collection")
collection = Collection(collection_name, schema)

# Insert some vectors
vectors = [
    [0.1, 0.2, 0.3, 0.4],
    [0.2, 0.1, 0.0, 0.9],
    [0.9, 0.1, 0.3, 0.2],
]

insert_result = collection.insert([vectors])
collection.flush()

print("Inserted IDs:", insert_result.primary_keys)

# Create index
index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {"M": 8, "efConstruction": 64},
}
collection.create_index("embedding", index_params)
collection.load()

# Search
query_vec = [0.1, 0.2, 0.3, 0.4]
search_params = {"metric_type": "L2", "params": {"ef": 64}}

results = collection.search(
    data=[query_vec],
    anns_field="embedding",
    param=search_params,
    limit=3,
)

for hits in results:
    for hit in hits:
        print(f"Hit id={hit.id}, distance={hit.distance}")
