"""Service layer for model loading and inference."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


# Base directory for model artifacts
ARTIFACTS_DIR = Path("/home/ubun/macromill/work/artifacts")

# Default model mapping
DEFAULT_MODEL = "tfidf_lr"

# Model metadata cache
MODEL_METADATA: dict[str, dict[str, Any]] = {}


@dataclass
class PreprocessConfig:
    """Preprocessing configuration."""
    strip_html: bool = True
    lowercase: bool = True


def _get_preprocess_cfg(strip_html: bool = True, lowercase: bool = True) -> PreprocessConfig:
    """Create preprocess config."""
    return PreprocessConfig(strip_html=strip_html, lowercase=lowercase)


def _clean_text(text: str, cfg: PreprocessConfig) -> str:
    """Clean text according to preprocessing config."""
    import html
    import re
    
    s = text
    if cfg.strip_html:
        s = html.unescape(s)
        s = re.sub(r"<[^>]+>", " ", s)
    if cfg.lowercase:
        s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


class ModelService:
    """Service for loading and running models."""
    
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self._models: dict[str, Any] = {}
        self._metadata: dict[str, dict] = {}
    
    def load_model(self, model_name: str) -> Any:
        """Load a model by name."""
        if model_name in self._models:
            return self._models[model_name]
        
        # Determine artifact path based on model name
        if model_name == "tfidf_lr":
            artifact_path = self.artifacts_dir / "tfidf_lr_v3"
        elif model_name == "tfidf_linearsvm":
            artifact_path = self.artifacts_dir / "tfidf_linearsvm_v3"
        elif model_name == "roberta":
            artifact_path = self.artifacts_dir / "roberta_v3"
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Load the model
        model = self._load_artifact(artifact_path)
        self._models[model_name] = model
        
        # Load metadata if available
        meta_path = artifact_path / "model_meta.json"
        if meta_path.exists():
            self._metadata[model_name] = json.loads(meta_path.read_text(encoding="utf-8"))
        
        return model
    
    def _load_artifact(self, path: Path) -> Any:
        """Load an artifact from disk."""
        import joblib
        
        # Try to load as joblib file first
        pkl_files = list(path.glob("*.pkl")) + list(path.glob("*.joblib"))
        if pkl_files:
            return joblib.load(pkl_files[0])
        
        # Check for PyTorch model
        if (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists():
            return self._load_roberta(path)
        
        raise ValueError(f"No model file found in {path}")
    
    def _load_roberta(self, path: Path) -> Any:
        """Load RoBERTa model."""
        from transformers import RobertaForSequenceClassification, RobertaTokenizer
        
        tokenizer = RobertaTokenizer.from_pretrained(str(path))
        model = RobertaForSequenceClassification.from_pretrained(str(path))
        model.eval()
        
        # Return a wrapper that handles both
        return {"tokenizer": tokenizer, "model": model}
    
    def predict(
        self,
        text: str,
        model_name: str = DEFAULT_MODEL,
        preprocess: bool = True,
    ) -> dict[str, Any]:
        """Run prediction on text."""
        # Load model if not already loaded
        model = self.load_model(model_name)
        
        # Preprocess if requested
        cfg = _get_preprocess_cfg(strip_html=preprocess, lowercase=preprocess)
        cleaned_text = _clean_text(text, cfg) if preprocess else text
        
        # Run inference
        start_time = time.perf_counter()
        
        if model_name == "roberta":
            result = self._predict_roberta(model, cleaned_text)
        else:
            result = self._predict_sklearn(model, cleaned_text)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "text": text,
            "cleaned_text": cleaned_text,
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "model_used": model_name,
            "preprocessing": preprocess,
            "latency_ms": latency_ms,
        }
    
    def _predict_sklearn(self, model: Any, text: str) -> dict[str, Any]:
        """Predict using sklearn model."""
        prediction = model.predict([text])[0]
        
        # Get probabilities if available
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([text])[0]
            confidence = float(max(proba))
        else:
            # If no predict_proba, assume 0.5 confidence
            confidence = 0.5
        
        # prediction is a string ('negative' or 'positive')
        sentiment = str(prediction)
        
        # Get probabilities mapping
        if proba is not None and hasattr(model, "classes_"):
            classes = model.classes_
            prob_dict = {str(c): float(p) for c, p in zip(classes, proba)}
        else:
            prob_dict = {
                "negative": 1.0 - confidence,
                "positive": confidence,
            }
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": prob_dict,
        }
    
    def _predict_roberta(self, model: Any, text: str) -> dict[str, Any]:
        """Predict using RoBERTa model."""
        import torch
        
        tokenizer = model["tokenizer"]
        roberta_model = model["model"]
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Predict
        with torch.no_grad():
            outputs = roberta_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
        
        # Get prediction
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
        
        sentiment = "positive" if pred_idx == 1 else "negative"
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": {
                "negative": probs[0].item(),
                "positive": probs[1].item(),
            }
        }
    
    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """Get information about a model."""
        # Load metadata if available
        if model_name not in self._metadata:
            self.load_model(model_name)  # This populates metadata
        
        meta = self._metadata.get(model_name, {})
        
        # Determine model type
        model_type = "transformer" if model_name == "roberta" else "tfidf"
        
        return {
            "name": model_name,
            "type": model_type,
            "accuracy": meta.get("accuracy"),
            "f1": meta.get("f1"),
            "latency_ms": meta.get("latency_ms"),
        }
    
    def list_models(self) -> list[str]:
        """List available model names."""
        return ["tfidf_lr", "tfidf_linearsvm", "roberta"]
    
    def get_default_model(self) -> str:
        """Get the default model name."""
        return DEFAULT_MODEL


# Global service instance
_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Get the global model service instance."""
    global _service
    if _service is None:
        _service = ModelService()
    return _service
