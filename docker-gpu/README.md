# Macromill Sentiment Analysis API - GPU Docker Deployment

A production-ready Docker container for the Macromill Sentiment Analysis API with GPU support. This image includes pre-trained models for sentiment classification and supports both inference and training workflows with CUDA acceleration.

> **Note**: The GPU image is optimized for training. For pure inference, consider using the lighter CPU image (`macromill/sentiment-api-cpu:latest`).

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Building the Image](#building-the-image)
- [Running the Container](#running-the-container)
- [API Usage](#api-usage)
- [Available Models](#available-models)
- [Configuration](#configuration)
- [Training](#training)
- [Health Checks](#health-checks)
- [Troubleshooting](#troubleshooting)

## Features

- **GPU Acceleration**: CUDA 11.6 support with cuDNN 8.4.1 for accelerated training
- **Multiple Models**: TF-IDF + Logistic Regression, TF-IDF + Linear SVM, RoBERTa Transformer
- **RESTful API**: FastAPI-based API with automatic documentation
- **Production Ready**: Includes health checks, configurable port, and health status endpoint
- **Pre-trained Artifacts**: Model artifacts bundled in the image
- **Training Support**: Can retrain models inside the container with GPU acceleration

## Prerequisites

- Docker 20.10+
- Docker Compose (optional, for easier management)
- NVIDIA GPU with CUDA support (for GPU training)
- nvidia-docker2 or NVIDIA Container Toolkit installed
- 18GB+ disk space for the image

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
cd docker-gpu
docker-compose up -d
```

### Option 2: Manual Build and Run

```bash
# Build the image
docker build -t macromill/sentiment-api-gpu:latest .

# Run the container
docker run -d -p 8000:8000 --name macromill-sentiment-api-gpu macromill/sentiment-api-gpu:latest
```

### Option 3: Run with GPU Support

```bash
# Run with GPU access (requires nvidia-docker2)
docker run -d \
  --gpus all \
  -p 8000:8000 \
  --name macromill-sentiment-api-gpu \
  macromill/sentiment-api-gpu:latest
```

## Building the Image

### Build Command

```bash
docker build -t macromill/sentiment-api-gpu:latest .
```

### Build with Custom Tag

```bash
docker build -t macromill/sentiment-api:v1.0.0 .
```

### Build Arguments

The image supports the following build arguments (if needed for future GPU builds):

```bash
docker build --build-arg PYTHON_VERSION=3.11 .
```

## Running the Container

### Basic Usage

```bash
docker run -d \
  -p 8000:8000 \
  --name macromill-sentiment-api-gpu \
  macromill/sentiment-api-gpu:latest
```

### With Custom Environment Variables

```bash
docker run -d \
  -p 8000:8000 \
  -e LOG_LEVEL=debug \
  --name macromill-sentiment-api-gpu \
  macromill/sentiment-api-gpu:latest
```

### Using Docker Compose

```bash
cd docker-gpu
docker-compose up -d
```

## API Usage

Once the container is running, you can access:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check and loaded models |
| `/predict` | POST | Run sentiment prediction |
| `/models` | GET | List available models |
| `/models/{model_name}` | GET | Get model details |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

### Example: Predict Sentiment

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This movie was absolutely amazing! I loved every moment of it."
  }'
```

**Response:**
```json
{
  "text": "This movie was absolutely amazing! I loved every moment of it.",
  "prediction": {
    "sentiment": "positive",
    "confidence": 0.946,
    "probabilities": {
      "negative": 0.054,
      "positive": 0.946
    }
  },
  "model_used": "tfidf_lr",
  "preprocessing": true
}
```

### Example: Specify Model

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This was the worst movie I have ever seen.",
    "model_name": "roberta"
  }'
```

### Check API Health

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": ["tfidf_lr", "tfidf_linearsvm", "roberta"]
}
```

## Available Models

| Model | Type | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `tfidf_lr` | TF-IDF + Logistic Regression | Fastest | Good | Production inference, high throughput |
| `tfidf_linearsvm` | TF-IDF + Linear SVM | Fast | Good | Balanced performance |
| `roberta` | RoBERTa Transformer | Slow | Highest | Best accuracy, lower throughput |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `info` | Logging level (debug, info, warning, error) |
| `WORKERS` | `1` | Number of Uvicorn workers |

### Port Mapping

The default port is `8000`. To change:

```bash
docker run -d -p 9000:8000 macromill/sentiment-api-gpu:latest
```

## Training

The container includes training functionality. To train a model inside the container:

```bash
# Train TF-IDF model
docker exec macromill-sentiment-api-gpu python -m macromill_sentiment.cli train --model tfidf_lr

# Train RoBERTa model (requires GPU for best performance)
docker exec macromill-sentiment-api-gpu python -m macromill_sentiment.cli train --model roberta
```

### Available Training Options

```bash
docker exec macromill-sentiment-api-gpu python -m macromill_sentiment.cli --help
```

## Health Checks

The container includes a built-in health check:

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' macromill-sentiment-api-gpu

# Check API health endpoint
curl http://localhost:8000/health
```

## Stopping the Container

```bash
# Using docker-compose
docker-compose down

# Or using docker
docker stop macromill-sentiment-api-gpu
docker rm macromill-sentiment-api-gpu
```

## Troubleshooting

### Container Won't Start

Check logs:
```bash
docker logs macromill-sentiment-api-gpu
```

### API Not Responding

Verify the container is running:
```bash
docker ps
```

Check port availability:
```bash
netstat -tuln | grep 8000
```

### Out of Memory

If running large models causes memory issues, try:
```bash
docker run -d -m=2g -p 8000:8000 macromill/sentiment-api-gpu:latest
```

### Rebuilding After Changes

```bash
# Stop and remove container
docker stop macromill-sentiment-api-gpu && docker rm macromill-sentiment-api-gpu

# Rebuild
docker build -t macromill/sentiment-api-gpu:latest .

# Run again
docker run -d -p 8000:8000 --name macromill-sentiment-api-gpu macromill/sentiment-api-gpu:latest
```

## GPU Configuration

### NVIDIA Container Toolkit Setup

For GPU support, ensure NVIDIA Container Toolkit is installed:

```bash
# Check if nvidia-docker is installed
nvidia-smi

# If not installed, follow instructions at:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Verify GPU Access in Container

```bash
# Check GPU is visible inside container
docker exec macromill-sentiment-api-gpu nvidia-smi

# Check PyTorch can see CUDA
docker exec macromill-sentiment-api-gpu python -c "import torch; print(torch.cuda.is_available())"
```

### Docker Compose with GPU

The docker-compose.yml already includes GPU configuration:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

## Image Specifications

| Component | Version |
|-----------|---------|
| Base Image | python:3.11-slim |
| Python | 3.11 |
| PyTorch | 1.13.1+cu116 |
| CUDA | 11.6 |
| cuDNN | 8.4.1 |
| Image Size | ~17.4GB |

## License

Proprietary - Macromill Internal Use Only
