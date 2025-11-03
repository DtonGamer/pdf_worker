
# PDF Processing Worker 🚀

A high-performance, memory-optimized Python worker for processing PDF documents and images with OCR support, text chunking, and semantic embeddings generation. Built for deployment on free-tier cloud platforms (Railway, Render, Zeabur).

## ✨ Features

- **📄 PDF Processing**: Extract text from digital PDFs and scanned documents
- **🖼️ Image OCR**: Process images (JPG, PNG) with Tesseract OCR
- **🧠 Semantic Embeddings**: Generate vector embeddings using BGE-small-en-v1.5
- **✂️ Smart Chunking**: Token-based text chunking with overlap for optimal context
- **🔄 Priority Queue System**: Support for high/normal/low priority job queues
- **♻️ Automatic Retry**: Exponential backoff retry logic for transient failures
- **💾 Efficient Storage**: Upsert-based database operations to handle reprocessing
- **🏥 Health Checks**: Built-in HTTP health endpoint for monitoring
- **📊 Memory Optimized**: Runs efficiently on 512MB RAM environments
- **🌐 Multi-Platform**: Deploy on Railway, Render, or Zeabur

## 🏗️ Architecture

```
User Upload → Supabase Edge Function → Redis Queue → Worker → PostgreSQL (pgvector)
                                           ↓
                                    PDF/Image Processing
                                           ↓
                                      Text Chunking
                                           ↓
                                    Embedding Generation
                                           ↓
                                    Database Storage
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Redis instance (Upstash recommended)
- Supabase project with PostgreSQL + pgvector
- Docker (for containerized deployment)

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/pdf-worker.git
cd pdf-worker
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your credentials
```

4. **Run the worker**
```bash
python worker.py
```

### Environment Variables

```bash
# Required
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_key
REDIS_URL=redis://your_redis_url

# Optional
BATCH_SIZE=4                    # Embedding batch size (default: 4)
CHUNK_SIZE=500                  # Tokens per chunk (default: 500)
CHUNK_OVERLAP=50                # Token overlap (default: 50)
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
OCR_LANGUAGES=eng               # Comma-separated (e.g., eng,spa,fra)
OCR_DPI=300                     # OCR resolution (default: 300)
PORT=8080                       # Health check port
WORKER_ID=worker-1              # Unique worker identifier
```

## 📦 Deployment

### Railway

1. Connect your GitHub repository to Railway
2. Add environment variables in Railway dashboard
3. Deploy using the included `railway.toml`

```toml
[build]
builder = "DOCKERFILE"

[deploy]
startCommand = "python -u worker.py"
```

**Health Check Configuration:**
- Path: `/health`
- Port: `8080`
- Timeout: `300s`

### Render

1. Create a new Web Service
2. Connect your repository
3. Use the included `Dockerfile`
4. Add environment variables
5. Configure health check path: `/health`

### Zeabur

1. Import from GitHub
2. Configure using `zeabur.toml`
3. Set environment variables
4. Health check: HTTP on port `${WEB_PORT}`, path `/health`

## 🗄️ Database Schema

### Documents Table
```sql
CREATE TABLE documents (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id),
  filename TEXT,
  storage_path TEXT,
  mime_type TEXT,
  file_size BIGINT,
  status TEXT,
  status_message TEXT,
  error_message TEXT,
  chunk_count INTEGER,
  needs_ocr BOOLEAN,
  processing_started_at TIMESTAMPTZ,
  processing_completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Knowledge Base Table (with pgvector)
```sql
CREATE TABLE knowledge_base (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  content TEXT NOT NULL,
  embedding VECTOR(384) NOT NULL,
  chunk_index INTEGER NOT NULL,
  token_count INTEGER NOT NULL,
  file_type TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  CONSTRAINT knowledge_base_document_chunk_unique UNIQUE (document_id, chunk_index)
);

-- Indexes
CREATE INDEX idx_knowledge_base_document_id ON knowledge_base(document_id);
CREATE INDEX idx_knowledge_base_user_id ON knowledge_base(user_id);
CREATE INDEX idx_knowledge_base_embedding ON knowledge_base 
  USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

## 📊 Performance

### Memory Usage
- **Idle**: ~400MB
- **Processing**: ~560-620MB
- **Peak (with embeddings)**: ~650MB

### Processing Speed
- **Digital PDF** (10 pages): ~5-10 seconds
- **Scanned PDF with OCR** (10 pages): ~30-60 seconds
- **Image OCR**: ~5-15 seconds per image

### Throughput
- **Documents/hour**: ~200-300 (depending on complexity)
- **Concurrent processing**: 1 document at a time (by design)
- **Queue capacity**: Unlimited (Redis-backed)

## 🔧 Configuration

### Embedding Models

Current: `BAAI/bge-small-en-v1.5` (384 dimensions)

Alternative models (edit `config.py`):
```python
# Smaller, faster
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # 384d, 23MB

# Larger, more accurate
EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'  # 768d, 430MB

# Multilingual
EMBEDDING_MODEL = 'BAAI/bge-m3'  # 1024d, 2.2GB, 100+ languages
```

### OCR Configuration

```python
# Single language
OCR_LANGUAGES = 'eng'

# Multiple languages
OCR_LANGUAGES = 'eng,spa,fra'

# Higher quality (slower)
OCR_DPI = 600
```

### Priority Queues

Jobs can be submitted to different priority queues:

```javascript
// High priority (processed first)
redis.rpush('pdf-processing-high', JSON.stringify(jobData));

// Normal priority
redis.rpush('pdf-processing', JSON.stringify(jobData));

// Low priority (background tasks)
redis.rpush('pdf-processing-low', JSON.stringify(jobData));
```

## 🔍 Monitoring

### Health Check Endpoint

```bash
# Check if worker is running
curl http://localhost:8080/health
# Response: "Worker is running"

# Get metrics
curl http://localhost:8080/metrics
```

**Metrics Response:**
```json
{
  "worker_id": "worker-3015",
  "jobs_processed": 42,
  "current_queue_length": 3,
  "uptime_seconds": 3600,
  "memory_usage_mb": 587.2,
  "timestamp": 1698765432.123
}
```

### Logs

The worker outputs structured logs for easy debugging:

```
🚀 Initializing PDF Worker [worker-3015]...
[worker-3015] ✓ Connected to Supabase
[worker-3015] ✓ Connected to Redis
[worker-3015] ✓ Listening to queue: pdf-processing
[worker-3015] 🎯 Worker ready. Waiting for jobs...

============================================================
[worker-3015] 📄 Processing document: a1d04bd4-3779-4980-8f60-6a4f7ce0dc45
[worker-3015]    Storage path: user-123/document.pdf
[worker-3015]    File type: application/pdf
[worker-3015]    Needs OCR: False
[worker-3015]    User ID: 37e1be6c-6c04-4307-9b6d-7ac218a4b6a7
============================================================

[worker-3015] ⬇️  Downloading file...
[worker-3015] ✓ Downloaded to: /tmp/pdf-processing/a1d04bd4.pdf
[worker-3015] 📄 Processing as PDF...
[worker-3015] ✓ Extracted 24915 characters (OCR: False)
[worker-3015] ✂️  Chunking text...
[worker-3015] ✓ Created 18 chunks
[worker-3015] 🧠 Generating embeddings...
[worker-3015]   Batch 1/5 complete
[worker-3015] ✓ Generated 18 embeddings
[worker-3015] 💾 Storing embeddings in database...
[worker-3015] ✓ Stored 18 chunks in database
[worker-3015] ✅ Successfully processed document
```

## 🐛 Troubleshooting

### Common Issues

**Worker keeps crashing with OOM errors**
- Reduce `BATCH_SIZE` to 2
- Reduce `CHUNK_SIZE` to 400
- Upgrade to a plan with more RAM

**OCR not working**
- Ensure Tesseract is installed in your Docker image
- Check `OCR_LANGUAGES` environment variable
- Verify OCR language packs are installed

**Embeddings generation is slow**
- Default batch size of 4 is optimized for 512MB RAM
- Increase `BATCH_SIZE` if you have more memory
- Consider using a smaller model like `all-MiniLM-L6-v2`

**Health checks failing**
- Ensure `PORT` environment variable matches platform requirements
- Zeabur uses `WEB_PORT`, Railway/Render use `PORT`
- Check that port 8080 is exposed in your Dockerfile

**Jobs not being processed**
- Verify Redis connection with `redis-cli PING`
- Check queue length with `redis-cli LLEN pdf-processing`
- Ensure Supabase credentials are correct

## 📚 Tech Stack

- **Language**: Python 3.11
- **Embeddings**: sentence-transformers, BGE-small-en-v1.5
- **OCR**: Tesseract, pytesseract, OpenCV
- **PDF Processing**: PyMuPDF, PyPDF2, ocrmypdf
- **Queue**: Redis (Upstash)
- **Database**: PostgreSQL with pgvector (Supabase)
- **Deployment**: Docker, Railway, Render, Zeabur

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [BAAI](https://github.com/FlagOpen/FlagEmbedding) for BGE embedding models
- [sentence-transformers](https://www.sbert.net/) for the embedding library
- [Supabase](https://supabase.com/) for the database infrastructure
- [Upstash](https://upstash.com/) for Redis hosting


---

Built with ❤️ for the RAG community
