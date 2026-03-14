"""API package for macromill sentiment analysis."""

from macromill_sentiment.api.main import app
from macromill_sentiment.api.service import ModelService, get_model_service
from macromill_sentiment.api.models import (
    PredictRequest,
    PredictResponse,
    PredictionResult,
    HealthResponse,
    ModelInfo,
    ModelsListResponse,
)

__all__ = [
    "app",
    "ModelService",
    "get_model_service",
    "PredictRequest",
    "PredictResponse",
    "PredictionResult",
    "HealthResponse",
    "ModelInfo",
    "ModelsListResponse",
]
