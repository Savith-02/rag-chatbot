# Milvus Schema Migration Guide

## Problem
The existing Milvus collection has an incompatible schema. Documents are missing the `source` field that the collection expects.

## Solution
We've implemented an explicit schema definition following Milvus best practices.

## Steps to Fix

### 1. Stop the running application
Press `Ctrl+C` in the terminal where uvicorn is running.

### 2. Drop the old collection (if it exists)
```bash
python -m app.manage_collection --drop
```
Type `yes` when prompted to confirm.

### 3. Create the new collection with proper schema
```bash
python -m app.manage_collection --create
```

This will create a collection with the following schema:
- **pk**: Primary key (auto-generated VARCHAR)
- **vector**: Float vector (1024 dimensions)
- **text**: Document text content (VARCHAR, max 65535 chars)
- **source**: Source file name (VARCHAR, max 512 chars)
- **file_name**: Original file name (VARCHAR, max 512 chars)
- **chunk_id**: Unique chunk identifier (VARCHAR, max 100 chars)
- **page_start**: Starting page number (INT64)
- **page_end**: Ending page number (INT64)
- **chunk_index**: Chunk position in document (INT64)
- **section_type**: Type of content (VARCHAR, max 50 chars)

### 4. Verify the collection
```bash
python -m app.manage_collection --info
```

This will show you the collection schema and current statistics.

### 5. Clear the processed files tracker
Since you're recreating the collection, you need to reprocess all files:
```bash
# On Windows (Git Bash)
rm processed_files.txt

# Or manually delete the file
```

### 6. Restart the application
```bash
python -m uvicorn app.main:app --reload
```

### 7. Trigger ingestion
Either wait for the scheduled task (10 minutes) or manually trigger:
```bash
curl -X POST http://localhost:8000/trigger_ingestion
```

## What Changed

### Before (Auto-generated schema)
- Schema was created automatically by langchain-milvus
- Inconsistent field definitions
- Missing required fields caused insertion errors

### After (Explicit schema)
- Schema is explicitly defined in `app/vectorstore.py`
- All fields are properly typed with constraints
- Collection is created with proper indexes
- `enable_dynamic_field=True` allows future flexibility

## Key Features

1. **Explicit Schema Definition**: Full control over field types and constraints
2. **Auto-creation**: Collection is created automatically on first use
3. **Index Configuration**: COSINE similarity with IVF_FLAT index
4. **Management Utility**: Easy collection management via CLI

## Schema Best Practices Applied

✅ **Primary Key**: Auto-generated VARCHAR primary key  
✅ **Vector Field**: Proper dimension specification (1024 for BGE-large-en)  
✅ **Text Fields**: Max length constraints to prevent errors  
✅ **Numeric Fields**: INT64 for page numbers and indices  
✅ **Index**: COSINE metric with IVF_FLAT for efficient search  
✅ **Dynamic Fields**: Enabled for future extensibility  

## Troubleshooting

### Collection already exists error
```bash
python -m app.manage_collection --recreate
```

### Schema mismatch error
Drop and recreate the collection:
```bash
python -m app.manage_collection --drop
python -m app.manage_collection --create
```

### Connection error
Ensure Milvus is running:
```bash
docker ps | grep milvus
```

## Future Enhancements

Consider these improvements:
- Use HNSW index for better performance (instead of IVF_FLAT)
- Add scalar indexes on frequently filtered fields (e.g., `file_name`)
- Implement partitioning by date or document type
- Add field-level compression for large text fields
