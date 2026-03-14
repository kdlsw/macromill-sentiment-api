from __future__ import annotations

from pathlib import Path


def load_imdb_csv(path: Path):
    import pandas as pd

    df = pd.read_csv(path)
    expected = {"review", "sentiment"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {sorted(missing)}")

    df = df[list(expected)].dropna()
    df["review"] = df["review"].astype(str)
    df["sentiment"] = df["sentiment"].astype(str)
    return df
