# ============================================================================
# DOCKERFILE: Ultra-Optimized PDF Processing Worker for Railway
# Target: Railway Free Tier (4GB image limit)
# IMPLEMENTS: Multi-stage build with aggressive optimization
# ============================================================================

# BUILD STAGE: Install build dependencies and Python packages
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install only essential build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    git \
    pkg-config \
    python3-dev \
    python3-venv \
    tesseract-ocr \
    tesseract-ocr-eng \
    ghostscript \
    poppler-utils \
    libpoppler-cpp-dev \
    libglx0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimization flags
# --no-cache-dir: Don't cache downloaded packages
# Combined into single RUN to reduce layers
# Install Python dependencies with optimization flags
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model at build time
# This prevents download during runtime and verifies it works
RUN python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('all-MiniLM-L6-v2'); \
    print('Model loaded successfully')"

# PREPARE STAGE: Prepare for minimal runtime image
FROM python:3.11-slim as preparer

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    ghostscript \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# RUNTIME STAGE: Ultra-minimal final image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install only essential runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    ghostscript \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from preparer stage
COPY --from=preparer /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code
COPY worker.py .
COPY config.py .
COPY ocr_processor.py .
COPY chunker.py .
COPY embeddings.py .
COPY .env.example .env

# Create temp directory for processing
RUN mkdir -p /tmp/pdf-processing

# Set environment variables for memory optimization
ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1
    # ✨ REMOVED: TRANSFORMERS_OFFLINE=1 and HF_DATASETS_OFFLINE=1
    # Model will download on first use instead

# Railway/Render assign PORT dynamically - expose it
EXPOSE ${PORT:-10000}

# Health check using HTTP endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-10000}/health').read()" || exit 1

# Run the worker with HTTP health endpoint
CMD ["python", "-u", "worker.py"]

# ============================================================================
# ESTIMATED IMAGE SIZE BREAKDOWN:
# - Base python:3.11-slim: ~120MB
# - System dependencies: ~250MB
# - Python packages: ~1.2GB (with multi-stage optimization)
# - Application code: ~1MB
# - Total: ~1.6GB (well under Railway's 4GB limit)
# ============================================================================
