# Migration Guide: Milvus Native Hybrid Search

## What Changed

Upgraded from manual BM25+RRF implementation to **Milvus native hybrid search** with built-in:
- ✅ BM25 sparse vectors (`SPARSE_FLOAT_VECTOR` field)
- ✅ Reciprocal Rank Fusion (`RRFRanker`)
- ✅ Inverted index for BM25 (`SPARSE_INVERTED_INDEX`)

## Migration Steps

### 1. Drop Existing Collection

**IMPORTANT**: The schema has changed. You must drop and recreate the collection.

```bash
# Option A: Via Python
python -c "from pymilvus import connections, utility; connections.connect(host='localhost', port='19530'); utility.drop_collection('financial_docs')"

# Option B: Via manage_collection.py (if you have it)
python app/manage_collection.py --drop
```

### 2. Clear Processed Files Tracker

Since you're re-ingesting everything:

```bash
# Windows
del processed_files.txt

# Linux/Mac
rm processed_files.txt
```

### 3. Restart Application

The new schema will be created automatically on startup:

```bash
# Stop current server (Ctrl+C)

# Start server
uvicorn app.main:app --reload
```

**Expected log output:**
```
Collection 'financial_docs' created successfully with hybrid search schema.
```

### 4. Re-ingest All Documents

Trigger ingestion to populate both dense and sparse vectors:

```bash
curl -X POST http://127.0.0.1:8000/trigger_ingestion
```

**Monitor logs** for:
```
Inserted X documents with hybrid vectors
```

## Verification

### Test Hybrid Search

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "revenue growth Q4 2023",
    "top_k": 5,
    "use_hybrid": true
  }'
```

**Check response** for:
```json
{
  "results": [
    {
      "search_method": "hybrid_milvus",
      ...
    }
  ]
}
```

### Test Dense-Only (Fallback)

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "revenue growth Q4 2023",
    "top_k": 5,
    "use_hybrid": false
  }'
```

## New Schema

```python
fields = [
    FieldSchema(name="pk", dtype=VARCHAR, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=FLOAT_VECTOR, dim=1024),          # Dense
    FieldSchema(name="sparse_vector", dtype=SPARSE_FLOAT_VECTOR),      # BM25 ← NEW
    FieldSchema(name="text", dtype=VARCHAR, max_length=65535),
    # ... metadata fields
]
```

## Key Benefits

1. **No external dependencies** - Removed `rank-bm25` package
2. **Faster** - BM25 computed inside Milvus (C++ implementation)
3. **Scalable** - Inverted index for efficient sparse search
4. **Native RRF** - Optimized fusion algorithm
5. **Simpler code** - ~200 lines removed from retrieval.py

## Troubleshooting

### Error: "Collection already exists"

Drop the collection first (see Step 1).

### Error: "Field 'sparse_vector' not found"

You're querying an old collection. Drop and recreate.

### Hybrid search returns empty results

Check logs for errors. Falls back to dense-only automatically.

### BM25 scores seem off

BM25 is fitted per-batch during ingestion. Re-ingest if you added many new documents.

## Rollback (if needed)

If you need to rollback to the previous implementation:

```bash
git checkout HEAD~1 -- app/retrieval.py app/vectorstore.py app/ingestion.py pyproject.toml
```

Then drop collection and re-ingest with old schema.
