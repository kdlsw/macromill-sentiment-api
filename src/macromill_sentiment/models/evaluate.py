from __future__ import annotations

import json
import time
from dataclasses import replace

from macromill_sentiment.artifacts.io import load_artifact
from macromill_sentiment.config import EvalConfig
from macromill_sentiment.data.load import load_imdb_csv
from macromill_sentiment.data.preprocess import clean_text, describe_preprocess
from macromill_sentiment.models.train import _normalize_label


def evaluate_model(cfg: EvalConfig) -> dict:
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    meta_source: str | None = None
    meta_path = cfg.artifact_path.with_suffix(cfg.artifact_path.suffix + ".meta.json")
    if cfg.use_artifact_meta and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if cfg.split_seed is None and "seed" in meta:
            cfg = replace(cfg, split_seed=int(meta["seed"]))
            meta_source = "artifact_meta"
        if cfg.test_size is None and "test_size" in meta:
            cfg = replace(cfg, test_size=float(meta["test_size"]))
            meta_source = "artifact_meta"
        if cfg.max_samples is None and meta.get("max_samples") is not None:
            cfg = replace(cfg, max_samples=int(meta["max_samples"]))
            meta_source = "artifact_meta"

    df = load_imdb_csv(cfg.data_path)
    if cfg.max_samples is not None and cfg.max_samples < len(df):
        seed_for_sampling = 42 if cfg.split_seed is None else int(cfg.split_seed)
        df = df.sample(n=int(cfg.max_samples), random_state=seed_for_sampling)
    X = df["review"].map(lambda s: clean_text(s, cfg.preprocess))
    y = df["sentiment"].map(_normalize_label).to_numpy()

    if cfg.split_seed is not None and cfg.test_size is not None:
        _, X, _, y = train_test_split(
            X,
            y,
            test_size=float(cfg.test_size),
            random_state=int(cfg.split_seed),
            stratify=y,
        )

    model = load_artifact(cfg.artifact_path)
    y_pred = np.asarray(model.predict(X))

    results: dict[str, object] = {
        "artifact_path": str(cfg.artifact_path),
        "preprocess": describe_preprocess(cfg.preprocess),
        "n_samples": int(len(y)),
    }
    if cfg.split_seed is not None and cfg.test_size is not None:
        results["split"] = {
            "seed": int(cfg.split_seed),
            "test_size": float(cfg.test_size),
            "source": meta_source or "cli",
        }

    # Get probability scores for ROC-AUC (if available)
    y_scores = None
    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        y_proba = model.predict_proba(X)
        # Get probability of positive class
        classes = getattr(model, "classes_", None)
        if classes is not None:
            pos_idx = list(classes).index("positive") if "positive" in classes else 1
            y_scores = y_proba[:, pos_idx]

    for m in cfg.metrics:
        if m == "accuracy":
            results["accuracy"] = float(accuracy_score(y, y_pred))
        elif m == "f1":
            results["f1"] = float(f1_score(y, y_pred, pos_label="positive"))
        elif m == "confusion_matrix":
            cm = confusion_matrix(y, y_pred, labels=["negative", "positive"])
            results["confusion_matrix"] = cm.tolist()
        elif m == "roc_auc":
            if y_scores is not None:
                # Convert labels to binary (0/1) for ROC-AUC
                y_binary = np.where(y == "positive", 1, 0)
                results["roc_auc"] = float(roc_auc_score(y_binary, y_scores))
            else:
                results["roc_auc"] = None
                results["warning"] = "ROC-AUC not available: model does not support predict_proba"
        else:
            raise ValueError(f"Unknown metric: {m!r}")

    # Performance metrics
    if cfg.measure_latency:
        # Warmup
        warmup_texts = X[: cfg.latency_warmup].tolist() if len(X) >= cfg.latency_warmup else X.tolist()
        for text in warmup_texts:
            _ = model.predict([text])

        # Timed iterations
        start_time = time.perf_counter()
        for _ in range(cfg.latency_n_iterations):
            _ = model.predict(X[:1].tolist())
        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_latency_ms = (total_time / cfg.latency_n_iterations) * 1000

        results["latency"] = {
            "avg_ms": round(avg_latency_ms, 4),
            "iterations": cfg.latency_n_iterations,
        }

        # Throughput (samples per second)
        # Estimate based on batch processing
        batch_size = min(100, len(X))
        start_time = time.perf_counter()
        _ = model.predict(X[:batch_size].tolist())
        end_time = time.perf_counter()
        batch_time = end_time - start_time
        throughput = batch_size / batch_time

        results["throughput"] = {
            "samples_per_sec": round(throughput, 2),
            "batch_size": batch_size,
        }

    # Artifact size
    artifact_size_bytes = cfg.artifact_path.stat().st_size
    results["artifact_size"] = {
        "bytes": artifact_size_bytes,
        "mb": round(artifact_size_bytes / (1024 * 1024), 2),
    }

    return results
