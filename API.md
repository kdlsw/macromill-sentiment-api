# Macromill Sentiment Analysis API

## Overview

**Base URL:** `http://localhost:8000`  
**API Version:** 1.0.0  
**Documentation:** `/docs` (Swagger UI), `/redoc` (ReDoc)

RESTful API for movie review sentiment classification using multiple machine learning models.

---

## Endpoints

### 1. Root

**GET** `/`

Redirects to API documentation.

**Response:**

```json
{
  "message": "Welcome to Macromill Sentiment Analysis API",
  "docs": "/docs",
  "redoc": "/redoc"
}
```

---

### 2. Health Check

**GET** `/health`

Check the health status of the API service.

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": ["tfidf_lr", "tfidf_linearsvm", "roberta"]
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Service status |
| `version` | string | API version |
| `models_loaded` | array | List of loaded model names |

---

### 3. Predict Sentiment

**POST** `/predict`

Predict sentiment of a movie review.

**Request Body:**

```json
{
  "text": "this is the best movie i have ever seen",
  "model_name": "tfidf_lr",
  "preprocess": true
}
```

**Request Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | Yes | - | The movie review text to analyze |
| `model_name` | string | No | `tfidf_lr` | Model to use: `tfidf_lr`, `tfidf_linearsvm`, or `roberta` |
| `preprocess` | boolean | No | `true` | Whether to preprocess text (strip HTML, lowercase) |

**Example Requests:**

```json
{
  "text": "this is the best movie i have ever seen"
}
```

```json
{
  "text": "Absolutely terrible film. Complete waste of time. The acting was horrible and the plot made no sense at all."
}
```

```json
{
  "text": "I loved every minute of this movie! Great storyline and fantastic performances by the cast."
}
```

```json
{
  "text": "Mediocre at best. Not terrible but certainly not worth watching again."
}
```

**Response:**

```json
{
  "text": "this is the best movie i have ever seen",
  "prediction": {
    "sentiment": "positive",
    "confidence": 0.95,
    "probabilities": {
      "negative": 0.05,
      "positive": 0.95
    }
  },
  "model_used": "tfidf_lr",
  "preprocessing": true
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | The input text (truncated if too long) |
| `prediction.sentiment` | string | Predicted sentiment: `positive` or `negative` |
| `prediction.confidence` | float | Prediction confidence score (0-1) |
| `prediction.probabilities` | object | Probability distribution over classes |
| `model_used` | string | Model used for this prediction |
| `preprocessing` | boolean | Whether preprocessing was applied |

**Error Responses:**

- `400 Bad Request` - Invalid model name
- `500 Internal Server Error` - Prediction failed

---

### 4. List Models

**GET** `/models`

List all available models with their information.

**Response:**

```json
{
  "models": [
    {
      "name": "tfidf_lr",
      "type": "tfidf",
      "accuracy": 0.89,
      "f1": 0.88,
      "latency_ms": 5.2
    },
    {
      "name": "tfidf_linearsvm",
      "type": "tfidf",
      "accuracy": 0.90,
      "f1": 0.89,
      "latency_ms": 8.1
    },
    {
      "name": "roberta",
      "type": "transformer",
      "accuracy": 0.93,
      "f1": 0.92,
      "latency_ms": 45.3
    }
  ],
  "default_model": "tfidf_lr"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `models` | array | List of available models |
| `models[].name` | string | Model identifier |
| `models[].type` | string | Model type: `tfidf` or `transformer` |
| `models[].accuracy` | float | Model accuracy on test set (if available) |
| `models[].f1` | float | Model F1 score on test set (if available) |
| `models[].latency_ms` | float | Average latency in milliseconds (if available) |
| `default_model` | string | Default model name |

---

### 5. Get Model Info

**GET** `/models/{model_name}`

Get information about a specific model.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | string | Model identifier: `tfidf_lr`, `tfidf_linearsvm`, or `roberta` |

**Response:**

```json
{
  "name": "tfidf_lr",
  "type": "tfidf",
  "accuracy": 0.89,
  "f1": 0.88,
  "latency_ms": 5.2
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Model identifier |
| `type` | string | Model type: `tfidf` or `transformer` |
| `accuracy` | float | Model accuracy on test set (if available) |
| `f1` | float | Model F1 score on test set (if available) |
| `latency_ms` | float | Average latency in milliseconds (if available) |

**Error Responses:**

- `404 Not Found` - Model not found

---

## Models

### Available Models

| Model Name | Type | Description | Accuracy | Latency |
|------------|------|-------------|----------|---------|
| `tfidf_lr` | TF-IDF + Logistic Regression | Fast, lightweight model | ~89% | ~5ms |
| `tfidf_linearsvm` | TF-IDF + Linear SVM | Higher accuracy, slightly slower | ~90% | ~8ms |
| `roberta` | RoBERTa Transformer | Best accuracy, GPU recommended | ~93% | ~45ms |

### Sentiment Classes

- **positive** - Positive movie review
- **negative** - Negative movie review

---

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Invalid model: invalid_model. Available: ['tfidf_lr', 'tfidf_linearsvm', 'roberta']"
}
```

### 404 Not Found

```json
{
  "detail": "Model not found: roberta. Available: ['tfidf_lr', 'tfidf_linearsvm']"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Prediction failed: <error_message>"
}
```

---

## Interactive Documentation

When the service is running, you can access:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

These provide interactive API testing and detailed schema documentation.
