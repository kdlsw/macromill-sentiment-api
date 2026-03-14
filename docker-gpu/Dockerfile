# Dockerfile for Macromill Sentiment Analysis API - GPU Version
# Production-ready container with FastAPI and CUDA 11.6 support

# Build stage - Python 3.10 to match host (Python 3.10.4)
FROM python:3.10-slim as builder

# NOTE: Exact dependency versions matching host (Python 3.10.4):
#   torch==1.13.1+cu116
#   torchvision==0.14.1+cu116
#   transformers==4.35.0
#   tokenizers==0.14.1
#   accelerate==1.13.0
#   tqdm==4.67.3
#   pandas==2.3.3
#   numpy==1.26.4
#   scikit-learn==1.7.2
#   joblib==1.5.3
#   fastapi==0.135.1
#   uvicorn==0.41.0
#   pydantic==2.12.5

# Install build dependencies
ENV http_proxy=
ENV https_proxy=
ENV HTTP_PROXY=
ENV HTTPS_PROXY=
ENV no_proxy=*

# Set pip timeout to 600 seconds (10 minutes) for slow downloads
ENV PIP_TIMEOUT=600
ENV PIP_READ_TIMEOUT=600
ENV PIP_CONNECT_TIMEOUT=30
# Using Tsinghua/Tuna mirror for regular Python packages (Chinese mirror - working well)
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA 11.6 support (must match host CUDA version)
# Using SJTU mirror for faster download in China
# Note: torchvision is needed for some operations
RUN pip install --no-cache-dir torch==1.13.1+cu116 torchvision==0.14.1+cu116 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu116

# Install remaining requirements - exact versions matching host
# Note: Install huggingface-hub first to avoid conflicts
RUN pip install --no-cache-dir \
    huggingface-hub==0.17.3 \
    tqdm==4.67.3 \
    pandas==2.3.3 \
    numpy==1.26.4 \
    scikit-learn==1.7.2 \
    joblib==1.5.3 \
    fastapi==0.135.1 \
    uvicorn==0.41.0 \
    pydantic==2.12.5

# Install transformers and its dependencies
RUN pip install --no-cache-dir \
    tokenizers==0.14.1 \
    transformers==4.35.0

# Install accelerate (compatible version with transformers 4.35.0)
RUN pip install --no-cache-dir \
    accelerate==0.20.0

# Re-install PyTorch to ensure correct version (in case pip upgraded it)
# Using SJTU mirror for faster download in China
RUN pip install --no-cache-dir --force-reinstall torch==1.13.1+cu116 torchvision==0.14.1+cu116 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu116

# Final stage - production image with Python 3.10
FROM python:3.10-slim

# NOTE: Exact dependency versions matching host (Python 3.10.4):
#   torch==1.13.1+cu116
#   torchvision==0.14.1+cu116
#   transformers==4.35.0
#   tokenizers==0.14.1
#   accelerate==1.13.0
#   tqdm==4.67.3
#   pandas==2.3.3
#   numpy==1.26.4
#   scikit-learn==1.7.2
#   joblib==1.5.3
#   fastapi==0.135.1
#   uvicorn==0.41.0
#   pydantic==2.12.5

# Install runtime dependencies
ENV http_proxy=
ENV https_proxy=
ENV HTTP_PROXY=
ENV HTTPS_PROXY=
ENV no_proxy=*

# Using Tsinghua/Tuna mirror (Chinese mirror - working well)
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Copy CUDA packages and installation script
COPY cuda-packages/ /tmp/cuda/
COPY install_cuda.sh /tmp/cuda/install_cuda.sh

# Install CUDA 11.6 and cuDNN
RUN chmod +x /tmp/cuda/install_cuda.sh && /tmp/cuda/install_cuda.sh

# Create venv and install dependencies from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy source code
COPY src/ /app/src/

# Set Python path to include the package (no need to install)
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Copy model artifacts
COPY artifacts/ /home/ubun/macromill/work/artifacts/

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["uvicorn", "macromill_sentiment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
