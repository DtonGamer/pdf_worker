# ============================================================================
# DOCKERFILE: Ultra-Optimized PDF Processing Worker for Railway
# ============================================================================

FROM python:3.11-slim as builder
WORKDIR /app

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
    # ✅ Add image and JP2 dependencies
    libjpeg-dev libtiff-dev libpng-dev libopenjp2-7 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('all-MiniLM-L6-v2'); \
    print('Model loaded successfully')"

# ---------------------------------------------------------------------------
FROM python:3.11-slim as preparer

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
    # ✅ Add image and JP2 dependencies
    libjpeg-dev libtiff-dev libpng-dev libopenjp2-7 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# ---------------------------------------------------------------------------
FROM python:3.11-slim
WORKDIR /app

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
    # ✅ Add image and JP2 dependencies
    libjpeg-dev libtiff-dev libpng-dev libopenjp2-7 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

COPY --from=preparer /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

COPY worker.py .
COPY config.py .
COPY ocr_processor.py .
COPY chunker.py .
COPY embeddings.py .
COPY .env.example .env

RUN mkdir -p /tmp/pdf-processing

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE ${PORT:-10000}

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-10000}/health').read()" || exit 1

CMD ["python", "-u", "worker.py"]

# ============================================================================
# ESTIMATED IMAGE SIZE BREAKDOWN:
# - Base python:3.11-slim: ~120MB
# - System dependencies: ~250MB
# - Python packages: ~1.2GB (with multi-stage optimization)
# - Application code: ~1MB
# - Total: ~1.6GB (well under Railway's 4GB limit)
# ============================================================================
