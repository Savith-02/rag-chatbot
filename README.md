# RAG Chatbot with Milvus

A Financial RAG (Retrieval-Augmented Generation) API that processes PDF documents, chunks them, embeds them using HuggingFace embeddings, and stores them in Milvus for semantic search.

## Features

- **Automated Folder Ingestion**: Automatically scans and processes PDF files from a designated folder every 10 minutes
- **File Upload**: Upload PDF files via API to be processed by the scheduled task
- **Manual Trigger**: Manually trigger ingestion on-demand via API
- **Processed File Tracking**: Remembers which files have been processed to avoid re-processing
- **Direct PDF Ingestion**: Upload and immediately process PDFs without saving to folder
- **Semantic Search**: Dense vector search using BGE-large-en embeddings for semantic similarity

## Setup

### Prerequisites

- Python 3.10+
- Milvus running locally (default: `localhost:19530`)

### Installation

1. Install dependencies:
```bash
uv sync
```

2. Start Milvus (if not already running):
```bash
# Using Docker
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest
```

3. Configure environment variables (optional):
```bash
# Create .env file
RAW_FILES_PATH=./raw_files  # Path to folder containing PDFs to process
PROCESSED_FILES_TRACKER=./processed_files.txt  # File tracking processed PDFs
```

### Running the Application

```bash
uv run --active uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Upload File to Raw Files Folder
Upload a PDF to the `raw_files` folder. It will be processed by the scheduled task (every 10 minutes).

```bash
curl -X POST http://localhost:8000/upload_file \
  -F "file=@/path/to/document.pdf"
```

Response:
```json
{
  "status": "success",
  "message": "File 'document.pdf' uploaded successfully",
  "file_path": "/path/to/raw_files/document.pdf",
  "note": "File will be processed by the scheduled ingestion task"
}
```

### 3. Trigger Manual Ingestion
Manually trigger the folder ingestion process to process all unprocessed PDFs immediately.

```bash
curl -X POST http://localhost:8000/trigger_ingestion
```

Response:
```json
{
  "status": "success",
  "timestamp": "2024-11-17T10:30:00.123456",
  "processed": 2,
  "skipped": 1,
  "failed": 0,
  "files_processed": [
    {
      "file_name": "report1.pdf",
      "chunks_indexed": 45
    },
    {
      "file_name": "report2.pdf",
      "chunks_indexed": 32
    }
  ],
  "files_skipped": ["already_processed.pdf"],
  "files_failed": []
}
```

### 4. Direct PDF Ingestion (Immediate Processing)
Upload and immediately process a PDF without saving to the raw_files folder.

```bash
curl -X POST http://localhost:8000/ingest_pdf \
  -F "file=@/path/to/document.pdf"
```

Response:
```json
{
  "file_name": "document.pdf",
  "chunks_indexed": 45
}
```

### 5. Query Documents
Search indexed documents using natural language queries with semantic vector search.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the financial projections for Q4?",
    "top_k": 5,
    "file_name": "report.pdf"
  }'
```

Response:
```json
{
  "results": [
    {
      "chunk_id": "report.pdf_5_2",
      "file_name": "report.pdf",
      "section_title": "Financial Results",
      "section_type": "text",
      "page_start": 5,
      "page_end": 5,
      "chunk_index": 2,
      "content": "Q4 financial projections show...",
      "score": 0.89,
      "search_method": "dense_only"
    }
  ]
}
```

### 6. Check Ingestion Status
Get the status of the scheduled ingestion job.

```bash
curl http://localhost:8000/ingestion_status
```

Response:
```json
{
  "status": "active",
  "job_id": "folder_ingestion_job",
  "next_run": "2024-11-17T10:40:00",
  "interval_minutes": 10
}
```

## How It Works

### Automated Ingestion Workflow

1. **Scheduled Task**: Every 10 minutes, the system scans the `raw_files` folder for PDF files
2. **File Tracking**: Checks `processed_files.txt` to see which files have already been processed
3. **Processing**: For each new file:
   - Extracts text from PDF pages
   - Chunks text into ~1000 character segments
   - Generates dense embeddings using `BAAI/bge-large-en`
   - Stores vectors in Milvus with metadata (file name, page number, chunk index)
4. **Tracking Update**: Marks processed files in `processed_files.txt`

### Search Workflow

1. **Query Processing**: User submits query
2. **Vector Generation**: Generate BGE embedding for query
3. **Milvus Search**: Execute semantic similarity search on dense vector field
4. **Results**: Return ranked results with metadata and scores

### File Upload Workflow

1. Upload file via `/upload_file` endpoint
2. File is saved to `raw_files` folder
3. Next scheduled task (within 10 minutes) will process it
4. Or manually trigger processing via `/trigger_ingestion`

## Configuration

### Environment Variables

- `RAW_FILES_PATH`: Directory path for raw PDF files (default: `./raw_files`)
- `PROCESSED_FILES_TRACKER`: Path to file tracking processed PDFs (default: `./processed_files.txt`)

### Milvus Configuration

Edit `app/vectorstore.py` to configure Milvus connection:

```python
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "financial_docs"
EMBEDDING_DIM = 1024  # BGE-large-en embedding dimension
```

### Milvus Collection Management

The collection schema is explicitly defined with the following fields:
- `pk`: Primary key (auto-generated)
- `vector`: Float vector (1024 dimensions for BGE-large-en)
- `text`: Document text content
- `source`: Source file name
- `file_name`: Original file name
- `chunk_id`: Unique chunk identifier
- `page_start`, `page_end`: Page numbers
- `chunk_index`: Chunk position in document
- `section_type`: Type of content (text/table/etc)

**Manage collection using the utility script:**

```bash
# Show collection information
python -m app.manage_collection --info

# Create collection with schema (if not exists)
python -m app.manage_collection --create

# Drop existing collection (WARNING: deletes all data)
python -m app.manage_collection --drop

# Drop and recreate collection (WARNING: deletes all data)
python -m app.manage_collection --recreate
```

## Project Structure

```
rag-chatbot/
├── app/
│   ├── config.py            # Configuration and paths
│   ├── ingestion.py         # PDF processing and folder ingestion
│   ├── main.py              # FastAPI app with scheduler
│   ├── manage_collection.py # Milvus collection management utility
│   ├── retrieval.py         # Query and search logic
│   └── vectorstore.py       # Milvus connection and schema
├── raw_files/               # Folder for PDFs to process (auto-created)
├── processed_files.txt      # Tracks processed files (auto-created)
├── pyproject.toml           # Dependencies
└── README.md
```

## Development

### Install Dependencies
```bash
uv sync
```

### Run Tests
```bash
pytest tests/
```

### API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI)

## Notes

- The scheduler runs in the background and processes files every 10 minutes
- Files are only processed once - tracked in `processed_files.txt`
- To reprocess a file, remove its entry from `processed_files.txt`
- The `/ingest_pdf` endpoint processes files immediately without tracking
- The `/upload_file` endpoint saves files for scheduled processing with tracking
