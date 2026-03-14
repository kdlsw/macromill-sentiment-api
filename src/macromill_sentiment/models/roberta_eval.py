"""
RoBERTa evaluation for sentiment analysis.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from macromill_sentiment.data.preprocess import clean_text


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
        
        return {"label": label, "scores": scores, "label_id": label_id}


def evaluate_roberta(
    data_path: Path,
    artifact_dir: Path,
    max_samples: int | None = None,
    batch_size: int = 32,
    measure_latency: bool = True,
    latency_warmup: int = 10,
    latency_n_iterations: int = 100,
    use_artifact_meta: bool = True,
    split_seed: int | None = None,
    test_size: float | None = None,
) -> dict:
    """Evaluate RoBERTa model on sentiment classification.
    
    Uses the held-out test set from training (via artifact metadata).
    """
    # Load metadata to get split information
    meta_path = artifact_dir / "model_meta.json"
    meta_source = None
    strip_html = True
    lowercase = True

    if use_artifact_meta and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

        if max_samples is None and "max_samples" in meta:
            max_samples = int(meta["max_samples"])
            meta_source = "artifact_meta"
        if split_seed is None and "seed" in meta:
            split_seed = int(meta["seed"])
            meta_source = "artifact_meta"
        if test_size is None and "test_size" in meta:
            test_size = float(meta["test_size"])
            meta_source = "artifact_meta"
        # Load preprocess settings from metadata
        if "preprocess" in meta:
            strip_html = meta["preprocess"].get("strip_html", True)
            lowercase = meta["preprocess"].get("lowercase", True)
    
    # Use defaults if not set
    if split_seed is None:
        split_seed = 42
    if test_size is None:
        test_size = 0.2

    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df = df.dropna()

    # Convert labels: positive=1, negative=0
    df["label"] = (df["sentiment"] == "positive").astype(int)

    # Apply max_samples limit BEFORE split (for consistency with training)
    if max_samples:
        df = df.sample(n=max_samples, random_state=split_seed).reset_index(drop=True)

    texts = df["review"].tolist()
    labels = df["label"].tolist()

    # Apply preprocessing (same as training)
    preprocess_cfg = type("PreprocessConfig", (), {"strip_html": strip_html, "lowercase": lowercase})()
    texts = [clean_text(t, preprocess_cfg) for t in texts]

    # Apply the same train/test split as training (to get the held-out test set)
    _, test_texts, _, test_labels = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=split_seed,
        stratify=labels,
    )

    print(f"Evaluating on held-out test set: {len(test_texts)} samples")
    print(f"  (max_samples={max_samples}, split_seed={split_seed}, test_size={test_size}, source={meta_source or 'cli'})")

    # Load predictor
    predictor = RoBERTaPredictor(artifact_dir)

    # Predict in batches on test set only
    predictions = []
    labels_out = []
    probabilities = []

    print("Running predictions...")
    
    for i in range(0, len(test_texts), batch_size):
        batch_texts = test_texts[i:i+batch_size]
        batch_labels = test_labels[i:i+batch_size]
        
        # Tokenize batch
        inputs = predictor.tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        inputs = {k: v.to(predictor.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = predictor.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(outputs.logits, dim=-1)
        
        predictions.extend(preds.cpu().tolist())
        labels_out.extend(batch_labels)
        probabilities.extend(probs[:, 1].cpu().tolist())  # Probability of positive class
        
        if (i + batch_size) % 1000 == 0 or i + batch_size >= len(test_texts):
            print(f"  Processed {min(i + batch_size, len(test_texts))}/{len(test_texts)} samples")

    # Calculate metrics
    acc = accuracy_score(labels_out, predictions)
    f1 = f1_score(labels_out, predictions, average="weighted")
    cm = confusion_matrix(labels_out, predictions)
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(labels_out, probabilities)
    except ValueError:
        roc_auc = None

    results = {
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "n_samples": len(test_texts),
        "model": "roberta-base",
        "preprocess": {"strip_html": strip_html, "lowercase": lowercase},
        "split": {
            "seed": split_seed,
            "test_size": test_size,
            "source": meta_source or "cli",
        },
    }

    # Performance metrics (using test set for measurement)
    if measure_latency and len(test_texts) > 0:
        # Warmup
        warmup_texts = test_texts[:latency_warmup] if len(test_texts) >= latency_warmup else test_texts
        for text in warmup_texts:
            _ = predictor.predict(text)

        # Single-sample latency
        start_time = time.perf_counter()
        for _ in range(latency_n_iterations):
            _ = predictor.predict(test_texts[0])
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_latency_ms = (total_time / latency_n_iterations) * 1000

        results["latency"] = {
            "avg_ms": round(avg_latency_ms, 4),
            "iterations": latency_n_iterations,
        }

        # Throughput (batch)
        eval_batch_size = min(100, len(test_texts))
        batch_texts = test_texts[:eval_batch_size]
        
        start_time = time.perf_counter()
        inputs = predictor.tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        inputs = {k: v.to(predictor.device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = predictor.model(**inputs)
        end_time = time.perf_counter()
        
        batch_time = end_time - start_time
        throughput = eval_batch_size / batch_time

        results["throughput"] = {
            "samples_per_sec": round(throughput, 2),
            "batch_size": eval_batch_size,
        }

    # Artifact size
    total_size = 0
    for f in artifact_dir.iterdir():
        if f.is_file():
            total_size += f.stat().st_size
    
    results["artifact_size"] = {
        "bytes": total_size,
        "mb": round(total_size / (1024 * 1024), 2),
    }

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--artifact-dir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    results = evaluate_roberta(
        data_path=Path(args.data_path),
        artifact_dir=Path(args.artifact_dir),
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )
    print(json.dumps(results, indent=2))
