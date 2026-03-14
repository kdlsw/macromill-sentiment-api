"""Pydantic models for API request and response."""

from typing import Optional, List
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request model for sentiment prediction."""
    text: str = Field(
        ..., 
        description="The movie review text to analyze",
        json_schema_extra={
            "examples": [
                {"text": "this is the best movie i have ever seen"},
                {"text": "Absolutely terrible film. Complete waste of time. The acting was horrible and the plot made no sense at all."},
                {"text": "I loved every minute of this movie! Great storyline and fantastic performances by the cast."},
                {"text": "Mediocre at best. Not terrible but certainly not worth watching again."},
            ]
        }
    )
    model_name: Optional[str] = Field(
        default="tfidf_lr",
        description="Model to use for prediction: tfidf_lr, tfidf_linearsvm, or roberta"
    )
    preprocess: Optional[bool] = Field(
        default=True,
        description="Whether to preprocess the text (strip HTML, lowercase)"
    )


class PredictionResult(BaseModel):
    """Single prediction result."""
    sentiment: str = Field(..., description="Predicted sentiment: positive or negative")
    confidence: float = Field(..., description="Prediction confidence score (0-1)")
    probabilities: Optional[dict] = Field(
        default=None,
        description="Probability distribution over classes"
    )


class PredictResponse(BaseModel):
    """Response model for sentiment prediction."""
    text: str = Field(..., description="The input text (truncated if too long)")
    prediction: PredictionResult = Field(..., description="Prediction result")
    model_used: str = Field(..., description="Model used for this prediction")
    preprocessing: bool = Field(..., description="Whether preprocessing was applied")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: List[str] = Field(..., description="List of loaded model names")


class ModelInfo(BaseModel):
    """Information about a single model."""
    name: str = Field(..., description="Model identifier")
    type: str = Field(..., description="Model type (tfidf or transformer)")
    accuracy: Optional[float] = Field(default=None, description="Model accuracy on test set")
    f1: Optional[float] = Field(default=None, description="Model F1 score on test set")
    latency_ms: Optional[float] = Field(default=None, description="Average latency in milliseconds")


class ModelsListResponse(BaseModel):
    """Response model for listing available models."""
    models: List[ModelInfo] = Field(..., description="List of available models")
    default_model: str = Field(..., description="Default model name")
