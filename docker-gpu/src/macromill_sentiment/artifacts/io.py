from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_artifact(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        import joblib

        joblib.dump(obj, tmp_path)
        tmp_path.replace(path)
        return
    except ImportError:
        pass
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    with tmp_path.open("wb") as f:
        pickle.dump(obj, f)
    tmp_path.replace(path)


def load_artifact(path: Path) -> Any:
    try:
        import joblib

        return joblib.load(path)
    except ImportError:
        pass

    with path.open("rb") as f:
        return pickle.load(f)
