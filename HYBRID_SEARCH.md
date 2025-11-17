# Hybrid Search Implementation for Financial Documents

## Overview

Implemented hybrid search using **Milvus native functionality** combining:
- **Dense vector search** (semantic understanding via BGE embeddings)
- **BM25 sparse search** (keyword matching via Milvus built-in BM25)
- **RRF (Reciprocal Rank Fusion)** (Milvus built-in ranker)

## Why Hybrid Search for Financial Documents?

Financial documents require both:
1. **Semantic understanding** - Understanding concepts like "revenue growth" or "market performance"
2. **Exact term matching** - Finding specific company names, financial metrics, dates, percentages

### Architecture

```
Query → [Milvus Collection]
     ├─→ Dense Vector Search (COSINE on vector field)
     ├─→ BM25 Sparse Search (BM25 on sparse_vector field)
     └─→ RRF Fusion (Milvus RRFRanker) → Top k results
```

**Key Advantage**: All processing happens inside Milvus - no external BM25 computation needed!

## Hyperparameters (Optimized for Finance)

### Search Weights
- **DENSE_WEIGHT = 0.6** (60%)
  - Prioritizes semantic similarity
  - Good for conceptual queries: "What was the company's financial performance?"
  
- **SPARSE_WEIGHT = 0.4** (40%)
  - Emphasizes exact keyword matching
  - Critical for: company names, ticker symbols, specific metrics, dates
  - Example: "AAPL Q4 2023 revenue"

### RRF Parameters
- **RRF_K = 60**
  - Controls how quickly rank importance decreases
  - Lower K = more weight on top results
  - 60 is balanced for financial docs (not too aggressive)

### Retrieval Strategy
- **Candidate Pool = 4 × top_k**
  - Retrieves 4x more candidates before fusion
  - Ensures good coverage from both methods
  - Example: If top_k=5, retrieves 20 candidates from each method

## Milvus Native BM25

Milvus automatically:
- Tokenizes text from the `text` field
- Builds inverted index with `SPARSE_INVERTED_INDEX`
- Computes BM25 scores using built-in BM25 metric
- No manual preprocessing needed!

**Schema**:
```python
# Dense vector field
FieldSchema(name="vector", dtype=FLOAT_VECTOR, dim=1024)

# Sparse vector field for BM25
FieldSchema(name="sparse_vector", dtype=SPARSE_FLOAT_VECTOR)
```

## API Usage

### Hybrid Search (Default)
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the revenue for Q4 2023?",
    "top_k": 5,
    "use_hybrid": true
  }'
```

### Dense-Only Search
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the revenue for Q4 2023?",
    "top_k": 5,
    "use_hybrid": false
  }'
```

### With File Filter
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "revenue growth",
    "top_k": 5,
    "file_name": "annual_report_2023.pdf",
    "use_hybrid": true
  }'
```

## Response Format

```json
{
  "results": [
    {
      "chunk_id": "report.pdf_5_2",
      "file_name": "report.pdf",
      "page_start": 5,
      "page_end": 5,
      "chunk_index": 2,
      "content": "Revenue for Q4 2023 was $1.2B...",
      "score": 0.0156,
      "search_method": "hybrid"
    }
  ]
}
```

## Tuning Guidelines

### When to Adjust Weights

**Increase DENSE_WEIGHT (0.7-0.8)** if:
- Users ask conceptual questions
- Synonyms and paraphrasing are common
- Documents use varied terminology

**Increase SPARSE_WEIGHT (0.5-0.6)** if:
- Users search for specific entities (names, codes, IDs)
- Exact term matching is critical
- Documents have standardized terminology

### When to Adjust RRF_K

**Lower K (30-50)** if:
- You want to prioritize top-ranked results more heavily
- Your dataset is small and precise

**Higher K (70-100)** if:
- You want more democratic fusion
- Your dataset is large and diverse

### When to Adjust Candidate Pool

**Increase multiplier (5x-6x)** if:
- You have many documents
- Results seem to miss relevant content

**Decrease multiplier (2x-3x)** if:
- Performance is slow
- You have fewer documents

## Performance Considerations

- **BM25 is computed in-memory** on the candidate pool (not the entire corpus)
- **Scales well** because BM25 only runs on dense search results
- **Typical latency**: +50-100ms compared to dense-only search

## Setup

### 1. Drop Existing Collection (if upgrading)

**Important**: The new schema includes a `sparse_vector` field. You must recreate the collection:

```python
# In Python or via manage_collection.py
from pymilvus import connections, utility

connections.connect(host="localhost", port="19530")
utility.drop_collection("financial_docs")
```

### 2. Restart Application

The collection will be recreated automatically with the new hybrid search schema.

### 3. Re-ingest Documents

Trigger ingestion to populate both dense and sparse vectors:
```bash
curl -X POST http://127.0.0.1:8000/trigger_ingestion
```

## Testing Recommendations

Compare hybrid vs dense-only on these query types:

1. **Exact entity queries**: "Apple Inc revenue"
2. **Conceptual queries**: "What was the financial performance?"
3. **Mixed queries**: "AAPL revenue growth trends"
4. **Numeric queries**: "25% increase in Q4"

Monitor the `search_method` field in responses to verify hybrid search is active.
