# macromill-sentiment-api

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sentiment classification for movie reviews using the IMDB dataset. This project implements multiple models ranging from traditional TF-IDF approaches to fine-tuned transformer models, with a FastAPI service for production deployment.

## Features

- **Multiple Models**: TF-IDF + Logistic Regression, TF-IDF + Linear SVM, and RoBERTa transformer
- **EDA Tools**: Comprehensive exploratory data analysis with visualizations
- **REST API**: FastAPI-based prediction service with Swagger documentation
- **CLI**: Command-line interface for training, evaluation, and prediction
- **Docker Support**: CPU-only and GPU (CUDA 11.6) containerized deployments

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/macromill-sentiment-api.git
cd macromill-sentiment-api

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Manual Model Download (Optional)

If you want to use the pre-trained RoBERTa safetensor v1.0 model, download it from the GitHub Releases assets and place it in the appropriate location:

- **Host machine**: `work/artifacts/roberta_v3/model.safetensors`
- **GPU Docker**: `docker-gpu/artifacts/roberta_v2/model.safetensors`
- **CPU Docker**: `docker-cpu/artifacts/roberta_v2/model.safetensors`

The API will automatically download the model on first run if not found locally.

### Running the API

```bash
# Start the FastAPI server
uvicorn macromill_sentiment.api.main:app --host 0.0.0.0 --port 8000

# Or use the wrapper script
python -m macromill_sentiment.cli serve
```

The API will be available at:
- **API**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Example Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic!", "model_name": "roberta_v3"}'
```

## Project Structure

```
macromill-sentiment-api/
├── src/macromill_sentiment/    # Main Python package
│   ├── api/                    # FastAPI service
│   ├── cli/                    # Command-line interface
│   ├── models/                 # Model implementations
│   ├── data/                   # Data loading and preprocessing
│   ├── analysis/               # EDA and visualization
│   └── artifacts/              # Model I/O utilities
├── work/                       # Local outputs
│   ├── eda/                    # EDA results
│   └── artifacts/              # Trained model artifacts
├── docker-cpu/                 # CPU-only Docker image
├── docker-gpu/                 # GPU Docker image (CUDA 11.6)
├── ENVIRONMENT.md              # Environment setup guide
├── EVALUATION.md               # Model evaluation results
└── README.md                   # This file
```

See [STRUCTURE.md](./STRUCTURE.md) for detailed structures.

## Models

| Model | Type | Accuracy | Latency | Size |
|-------|------|----------|---------|------|
| TF-IDF + LR | Traditional ML | 86.20% | 0.70ms | 1.36 MB |
| TF-IDF + Linear SVM | Traditional ML | 86.50% | 0.74ms | 1.36 MB |
| RoBERTa-base | Transformer | **92.15%** | 12.73ms | 478.72 MB |

See [EVALUATION.md](./EVALUATION.md) for detailed metrics. 

## Usage

### Exploratory Data Analysis

```bash
# Run EDA
python work/run_eda.py eda \
  --data-path "work/IMDB Dataset.csv" \
  --output-dir "work/eda"

# Compare model evaluations
python work/run_eda.py compare
```

### Training

```bash
# Train TF-IDF + Logistic Regression
python -m macromill_sentiment.cli train \
  --model tfidf_lr \
  --artifact-dir work/artifacts/tfidf_lr_v3

# Train RoBERTa (requires GPU)
python -m macromill_sentiment.cli train \
  --model roberta \
  --artifact-dir work/artifacts/roberta_v3 \
  --epochs 3 \
  --batch-size 16
```

### Evaluation

```bash
# Evaluate a model
python -m macromill_sentiment.cli eval \
  --artifact-path work/artifacts/tfidf_lr_v3/model.joblib \
  --output-json work/artifacts/tfidf_lr_v3/eval.json
```

### Local Prediction

```bash
# Predict sentiment
python -m macromill_sentiment.cli predict-local \
  --artifact-path work/artifacts/tfidf_lr_v3/model.joblib \
  --text "This film was terrible and boring."
```

## Docker Deployment

### CPU-Only

```bash
cd docker-cpu
docker build -t macromill/sentiment-api-cpu:latest .
docker run -d -p 8000:8000 --name macromill-api-cpu macromill/sentiment-api-cpu:latest
```

### GPU (CUDA 11.6)

```bash
cd docker-gpu
# CUDA packages are already in cuda-packages/ folder
docker build -t macromill/sentiment-api-gpu:latest .
docker run --gpus all -d -p 8000:8000 --name macromill-api-gpu macromill/sentiment-api-gpu:latest
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and loaded models status |
| `/predict` | POST | Sentiment prediction |
| `/models` | GET | List available models with metrics |
| `/docs` | GET | Swagger API documentation |

## CLI Reference

### `train`

```bash
python -m macromill_sentiment.cli train [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `tfidf_lr` | Model type: `tfidf_lr`, `tfidf_linearsvm`, `roberta` |
| `--artifact-dir` | (required) | Output directory for model artifacts |
| `--data-path` | `work/IMDB Dataset.csv` | Training data path |
| `--epochs` | `1` | Number of training epochs (RoBERTa only) |
| `--batch-size` | `16` | Batch size (RoBERTa only) |
| `--lr` | `2e-5` | Learning rate (RoBERTa only) |
| `--strip-html` | `True` | Strip HTML tags during preprocessing |
| `--lowercase` | `True` | Convert text to lowercase |

### `eval`

```bash
python -m macromill_sentiment.cli eval [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--artifact-path` | (required) | Path to model artifact |
| `--output-json` | (required) | Output JSON file path |
| `--data-path` | `work/IMDB Dataset.csv` | Evaluation data path |
| `--measure-latency` | `True` | Measure inference latency |

## Dataset

This project uses the [IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data) from Kaggle.

- **Size**: 50,000 reviews
- **Labels**: Positive / Negative
- **Format**: CSV with `review` and `sentiment` columns

See [DATA.md](./DATA.md) for detailed analysis. 

## Preprocessing

Text preprocessing is shared across EDA, training, and inference:

- **HTML**: Unescape entities and strip tags (e.g., `<br />`)
- **Lowercase**: Optional (enabled by default)
- **Whitespace**: Normalize repeated whitespace and trim

Implementation: `src/macromill_sentiment/data/preprocess.py`

## Requirements

### Development

- Python 3.10+
- CUDA 11.6+ (for GPU training)

### Runtime

See `requirements.txt` for the full list of dependencies.

### Docker

- CPU: `docker-cpu/` - python:3.11-slim
- GPU: `docker-gpu/` - python:3.11-slim + CUDA 11.6

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the RoBERTa model
- [IMDb Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data) for the dataset
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
