from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from macromill_sentiment.artifacts.io import load_artifact
from macromill_sentiment.config import ArtifactConfig, EvalConfig, PreprocessConfig, TrainConfig
from macromill_sentiment.data.preprocess import clean_text
from macromill_sentiment.models.evaluate import evaluate_model
from macromill_sentiment.models.registry import list_models
from macromill_sentiment.models.train import train_model
from macromill_sentiment.models.roberta_train import train_roberta
from macromill_sentiment.models.roberta_predict import RoBERTaPredictor


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "train":
        if args.model == "roberta":
            # RoBERTa uses its own training pipeline
            train_roberta(
                data_path=args.data_path,
                artifact_dir=args.artifact_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                max_length=args.max_length,
                max_samples=args.max_samples,
                seed=args.seed,
                test_size=args.test_size,
                local_files_only=args.local_files_only,
                strip_html=args.strip_html,
                lowercase=args.lowercase,
            )
            print(str(args.artifact_dir / "model_meta.json"))
            return 0

        cfg = TrainConfig(
            data_path=args.data_path,
            artifact=ArtifactConfig(output_dir=args.artifact_dir),
            preprocess=PreprocessConfig(strip_html=args.strip_html, lowercase=args.lowercase),
            model_name=args.model,
            seed=args.seed,
            max_samples=args.max_samples,
            test_size=args.test_size,
        )
        model_path = train_model(cfg)
        print(str(model_path))
        return 0

    if args.command == "eval":
        # Check if it's a RoBERTa model (has config.json in parent dir)
        artifact_dir = args.artifact_path.parent
        # Check for either pytorch_model.bin or model.safetensors
        has_roberta = (artifact_dir / "config.json").exists() and (
            (artifact_dir / "pytorch_model.bin").exists() or (artifact_dir / "model.safetensors").exists()
        )
        if has_roberta:
            # RoBERTa model
            from macromill_sentiment.models.roberta_eval import evaluate_roberta
            results = evaluate_roberta(
                data_path=args.data_path,
                artifact_dir=artifact_dir,
                max_samples=args.max_samples,
                measure_latency=args.measure_latency,
                latency_warmup=args.latency_warmup,
                latency_n_iterations=args.latency_iterations,
                use_artifact_meta=not args.ignore_meta,
                split_seed=args.seed,
                test_size=args.test_size,
            )
        else:
            cfg = EvalConfig(
                data_path=args.data_path,
                artifact_path=args.artifact_path,
                preprocess=PreprocessConfig(strip_html=args.strip_html, lowercase=args.lowercase),
                metrics=tuple(args.metrics),
                max_samples=args.max_samples,
                split_seed=args.seed,
                test_size=args.test_size,
                use_artifact_meta=not args.ignore_meta,
                measure_latency=args.measure_latency,
                latency_warmup=args.latency_warmup,
                latency_n_iterations=args.latency_iterations,
            )
            results = evaluate_model(cfg)
        out = json.dumps(results, indent=2, sort_keys=True)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(out + "\n", encoding="utf-8")
        else:
            print(out)
        return 0

    if args.command == "predict-local":
        # Check if it's a RoBERTa model
        artifact_dir = args.artifact_path.parent
        if (artifact_dir / "config.json").exists():
            # RoBERTa model
            predictor = RoBERTaPredictor(artifact_dir)
            pred = predictor.predict(args.text)
            resp = {"label": pred["label"]}
            if "scores" in pred:
                resp["scores"] = pred["scores"]
        else:
            model = load_artifact(args.artifact_path)
            cfg = PreprocessConfig(strip_html=args.strip_html, lowercase=args.lowercase)
            text = clean_text(args.text, cfg)
            pred = model.predict([text])[0]
            resp = {"label": str(pred)}
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([text])[0]
                classes = getattr(model, "classes_", None)
                if classes is not None and len(classes) == len(proba):
                    resp["scores"] = {str(c): float(p) for c, p in zip(classes, proba)}
        print(json.dumps(resp, indent=2, sort_keys=True))
        return 0

    raise RuntimeError(f"Unknown command: {args.command!r}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="macromill-sentiment")
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--data-path", type=Path, default=Path("work") / "IMDB Dataset.csv")
    train.add_argument("--artifact-dir", type=Path, required=True)
    train.add_argument("--model", type=str, default="tfidf_lr", choices=list_models())
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max-samples", type=int, default=None)
    train.add_argument("--test-size", type=float, default=0.2)

    # RoBERTa-specific arguments
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--lr", type=float, default=2e-5)
    train.add_argument("--max-length", type=int, default=256)
    train.add_argument("--local-files-only", action="store_true", default=True,
                        help="Only load models from local cache, don't attempt to download from HuggingFace")

    _add_preprocess_args(train)

    ev = sub.add_parser("eval")
    ev.add_argument("--data-path", type=Path, default=Path("work") / "IMDB Dataset.csv")
    ev.add_argument("--artifact-path", type=Path, required=True)
    ev.add_argument("--metrics", nargs="+", default=["accuracy", "f1", "confusion_matrix", "roc_auc"])
    ev.add_argument("--output-json", type=Path, default=None)
    ev.add_argument("--max-samples", type=int, default=None)
    ev.add_argument("--seed", type=int, default=None)
    ev.add_argument("--test-size", type=float, default=None)
    ev.add_argument("--ignore-meta", action="store_true", default=False)
    # Performance metrics arguments
    ev.add_argument("--measure-latency", action=argparse.BooleanOptionalAction, default=True)
    ev.add_argument("--latency-warmup", type=int, default=10)
    ev.add_argument("--latency-iterations", type=int, default=100)
    _add_preprocess_args(ev)

    pred = sub.add_parser("predict-local")
    pred.add_argument("--artifact-path", type=Path, required=True)
    pred.add_argument("--text", type=str, required=True)
    _add_preprocess_args(pred)

    return p


def _add_preprocess_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--strip-html", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lowercase", action=argparse.BooleanOptionalAction, default=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
