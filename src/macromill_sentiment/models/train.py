from __future__ import annotations

import json
from pathlib import Path

from macromill_sentiment.artifacts.io import save_artifact
from macromill_sentiment.config import TrainConfig
from macromill_sentiment.data.load import load_imdb_csv
from macromill_sentiment.data.preprocess import clean_text
from macromill_sentiment.models.registry import build_model


def train_model(cfg: TrainConfig) -> Path:
    from sklearn.model_selection import train_test_split

    df = load_imdb_csv(cfg.data_path)
    if cfg.max_samples is not None and cfg.max_samples < len(df):
        df = df.sample(n=int(cfg.max_samples), random_state=int(cfg.seed))
    X = df["review"].map(lambda s: clean_text(s, cfg.preprocess))
    y = df["sentiment"].map(_normalize_label)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(cfg.test_size),
        random_state=int(cfg.seed),
        stratify=y,
    )

    model = build_model(cfg.model_name)
    model.fit(X_train, y_train)

    save_artifact(model, cfg.artifact.model_path)
    meta_path = cfg.artifact.model_path.with_suffix(cfg.artifact.model_path.suffix + ".meta.json")
    meta = {
        "data_path": str(cfg.data_path),
        "model_name": cfg.model_name,
        "seed": int(cfg.seed),
        "test_size": float(cfg.test_size),
        "max_samples": None if cfg.max_samples is None else int(cfg.max_samples),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "preprocess": {"strip_html": bool(cfg.preprocess.strip_html), "lowercase": bool(cfg.preprocess.lowercase)},
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cfg.artifact.model_path


def _normalize_label(label: str) -> str:
    s = str(label).strip().lower()
    if s in {"pos", "positive", "1", "true"}:
        return "positive"
    if s in {"neg", "negative", "0", "false"}:
        return "negative"
    return s
