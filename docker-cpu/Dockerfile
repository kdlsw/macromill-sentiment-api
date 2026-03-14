# Dockerfile for Macromill Sentiment Analysis API
# Production-ready container with FastAPI

# Build stage
FROM python:3.11-slim as builder

# Install build dependencies
ENV http_proxy=
ENV https_proxy=
ENV HTTP_PROXY=
ENV HTTPS_PROXY=
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/requirements.txt

# Final stage - production image
FROM python:3.11-slim

# Install runtime dependencies
ENV http_proxy=
ENV https_proxy=
ENV HTTP_PROXY=
ENV HTTPS_PROXY=
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

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
