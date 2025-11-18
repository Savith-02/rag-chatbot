# RAG Chatbot with Milvus

Financial RAG API that processes PDF documents and enables semantic search using HuggingFace embeddings and Milvus vector database.

## Functionality

- **PDF Processing**: Automated and manual PDF ingestion with text chunking
- **Vector Storage**: BGE-large-en embeddings stored in Milvus
- **Semantic Search**: Natural language queries with similarity search
- **File Tracking**: Prevents duplicate processing of documents

## Usage

```bash
# Install dependencies
uv sync

# Start Milvus (Docker)
docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest

# Run application
uv run --active uvicorn app.main:app --reload
```

API available at `http://localhost:8000`

## API Endpoints

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### Upload File
```bash
curl -X POST http://localhost:8000/upload_file -F "file=@document.pdf"
```

### Trigger Manual Ingestion
```bash
curl -X POST http://localhost:8000/trigger_ingestion
```

### Direct PDF Ingestion
```bash
curl -X POST http://localhost:8000/ingest_pdf -F "file=@document.pdf"
```

### Query Documents
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "financial projections", "top_k": 5}'
```

### Chat with RAG
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "financial projections", "top_k": 5}'
```

### Ingestion Status
```bash
curl http://localhost:8000/ingestion_status
```

**API Documentation**: `http://localhost:8000/docs`
