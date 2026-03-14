"""FastAPI application for sentiment classification API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from macromill_sentiment.api.models import (
    HealthResponse,
    ModelInfo,
    ModelsListResponse,
    PredictRequest,
    PredictResponse,
    PredictionResult,
)
from macromill_sentiment.api.service import ModelService, get_model_service


# API version
API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - startup and shutdown."""
    # Startup: initialize model service
    service = get_model_service()
    # Preload default model
    try:
        service.load_model(service.get_default_model())
        print(f"Loaded default model: {service.get_default_model()}")
    except Exception as e:
        print(f"Warning: Failed to preload model: {e}")
    
    yield
    
    # Shutdown: cleanup if needed
    print("Shutting down API...")


# Create FastAPI application
app = FastAPI(
    title="Macromill Sentiment Analysis API",
    description="RESTful API for movie review sentiment classification",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - redirect to docs."""
    return {
        "message": "Welcome to Macromill Sentiment Analysis API",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    service = get_model_service()
    
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        models_loaded=service.list_models(),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Predict sentiment of a movie review.
    
    - **text**: The movie review text to analyze (required)
    - **model_name**: Model to use - tfidf_lr, tfidf_linearsvm, or roberta (default: tfidf_lr)
    - **preprocess**: Whether to preprocess text (strip HTML, lowercase) - default: true
    """
    service = get_model_service()
    
    # Validate model name
    available_models = service.list_models()
    model_name = request.model_name or service.get_default_model()
    
    if model_name not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {model_name}. Available: {available_models}",
        )
    
    # Run prediction
    try:
        result = service.predict(
            text=request.text,
            model_name=model_name,
            preprocess=request.preprocess,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    # Build response
    # Truncate text for display if too long
    display_text = result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"]
    
    return PredictResponse(
        text=display_text,
        prediction=PredictionResult(
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
        ),
        model_used=result["model_used"],
        preprocessing=result["preprocessing"],
    )


@app.get("/models", response_model=ModelsListResponse, tags=["Models"])
async def list_models():
    """
    List all available models.
    
    Returns information about each available model including:
    - name: Model identifier
    - type: Model type (tfidf or transformer)
    - accuracy: Model accuracy on test set (if available)
    - f1: Model F1 score (if available)
    - latency_ms: Average latency (if available)
    """
    service = get_model_service()
    
    models_info = []
    default_model = service.get_default_model()
    
    for model_name in service.list_models():
        try:
            info = service.get_model_info(model_name)
            models_info.append(
                ModelInfo(
                    name=info["name"],
                    type=info["type"],
                    accuracy=info.get("accuracy"),
                    f1=info.get("f1"),
                    latency_ms=info.get("latency_ms"),
                )
            )
        except Exception as e:
            # If model info fails, still include basic info
            models_info.append(
                ModelInfo(
                    name=model_name,
                    type="transformer" if model_name == "roberta" else "tfidf",
                )
            )
    
    return ModelsListResponse(
        models=models_info,
        default_model=default_model,
    )


@app.get("/models/{model_name}", response_model=ModelInfo, tags=["Models"])
async def get_model(model_name: str):
    """
    Get information about a specific model.
    
    - **model_name**: Model identifier (tfidf_lr, tfidf_linearsvm, or roberta)
    """
    service = get_model_service()
    
    if model_name not in service.list_models():
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {model_name}. Available: {service.list_models()}",
        )
    
    try:
        info = service.get_model_info(model_name)
        return ModelInfo(
            name=info["name"],
            type=info["type"],
            accuracy=info.get("accuracy"),
            f1=info.get("f1"),
            latency_ms=info.get("latency_ms"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
