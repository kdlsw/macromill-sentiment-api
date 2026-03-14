# macromill — Environment Description

## 1. Development Environment

- **OS**: Linux (Ubuntu-based)
- **CUDA**: 11.6 (for GPU training)
- **Python**: 3.10+ (virtualenv under `.venv`)

### Core Runtime Dependencies

| Package | Version | Description |
|---------|---------|-------------|
| numpy | 1.26.4 | Numerical computing |
| pandas | >=2.0.0 | Data manipulation |
| scikit-learn | >=1.7.0 | ML algorithms |
| joblib | >=1.5.0 | Model serialization |
| matplotlib | >=3.10.0 | Visualization |

### Transformer Dependencies (GPU)

| Package | Version | Description |
|---------|---------|-------------|
| torch | 1.13.1+cu116 | Deep learning framework |
| transformers | 4.35.0 | Hugging Face transformers |
| tokenizers | 0.14.1 | Fast tokenization |
| accelerate | 1.13.0 | Distributed training |
| datasets | 2.14.0 | Hugging Face datasets |
| huggingface-hub | 0.17.3 | Model hub client |

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# Install dependencies
pip install -r requirements.txt
```

## 2. Docker Deployment

### CPU-Only Image

Located in `docker-cpu/`:

```bash
cd docker-cpu
docker build -t macromill/sentiment-api-cpu:latest .
docker run -d -p 8000:8000 --name macromill-api-cpu macromill/sentiment-api-cpu:latest
```

### GPU Image (CUDA 11.6)

Located in `docker-gpu/`:

```bash
cd docker-gpu
bash ./download-cuda-debs.sh  # Download CUDA .deb files
docker build -t macromill/sentiment-api-gpu:latest .
docker run --gpus all -d -p 8000:8000 --name macromill-api-gpu macromill/sentiment-api-gpu:latest
```

### Docker Quick Reference

| Image | Base | Use Case |
|-------|------|----------|
| `docker-cpu/` | python:3.11-slim | Production inference (CPU) |
| `docker-gpu/` | python:3.11-slim + CUDA 11.6 | Training & GPU inference |

### Health Check

After starting the container:

- **API**: http://localhost:8000
- **Health**: http://localhost:8000/health
- **Docs**: http://localhost:8000/docs
