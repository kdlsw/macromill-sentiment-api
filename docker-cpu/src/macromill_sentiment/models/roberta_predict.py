"""
RoBERTa inference for sentiment analysis.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RoBERTaPredictor:
    """RoBERTa sentiment predictor."""

    def __init__(self, artifact_dir: Path, device: str | None = None):
        self.artifact_dir = Path(artifact_dir)
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        print(f"Loading RoBERTa model from {artifact_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(artifact_dir))
        self.model.to(self.device)
        self.model.eval()
        
        # Load metadata
        meta_path = self.artifact_dir / "model_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

    def predict(self, text: str) -> dict:
        """Predict sentiment for a single text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
        label_id = torch.argmax(probs, dim=-1).item()
        label = "positive" if label_id == 1 else "negative"
        scores = {
            "negative": probs[0][0].item(),
            "positive": probs[0][1].item(),
        }
        
        return {"label": label, "scores": scores}

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Predict sentiment for a batch of texts."""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


def load_roberta_predictor(artifact_dir: Path) -> RoBERTaPredictor:
    """Load RoBERTa predictor from artifact directory."""
    return RoBERTaPredictor(artifact_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    predictor = RoBERTaPredictor(Path(args.artifact_dir))
    result = predictor.predict(args.text)
    print(json.dumps(result, indent=2))
