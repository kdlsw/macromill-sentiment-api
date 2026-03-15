import argparse
import json
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from macromill_sentiment.analysis.eda import main as eda_main
from macromill_sentiment.analysis.eval_plots import plot_metrics_bar, generate_eval_summary, print_metrics_table


def eval_plot_main(argv: list[str] | None = None) -> int:
    """Generate evaluation plots and summaries from multiple model evaluation JSON files."""
    args = _build_eval_plot_parser().parse_args(argv)

    if args.eval_json is None:
        print("Error: At least one --eval-json argument is required")
        return 1

    eval_results = {}
    for eval_path in args.eval_json:
        eval_path = Path(eval_path)
        if not eval_path.exists():
            print(f"Warning: {eval_path} not found, skipping")
            continue

        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Use parent directory name as model name (e.g., "tfidf_lr_v3" from "eval.json")
        model_name = eval_path.parent.name
        eval_results[model_name] = data

    if not eval_results:
        print("Error: No valid evaluation JSON files found")
        return 1

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate text table (printed to console unless disabled)
    if args.text:
        table_output = print_metrics_table(eval_results)
        print("\n" + table_output + "\n")

    # Generate bar chart
    if args.plot:
        plot_metrics_bar(eval_results, out_dir / "metrics_bar.png")

    # Generate JSON summary
    if args.summary:
        generate_eval_summary(eval_results, out_dir / "summary.json")

    print(f"Generated evaluation summary in {out_dir}")
    return 0


def compare_models_main(artifacts_dir: Path, output_dir: Path, output_txt: Path | None = None) -> int:
    """Compare evaluation results across all models."""
    model_dirs = ["tfidf_lr_v3", "tfidf_linearsvm_v3", "roberta_v3"]

    eval_results: dict[str, dict[str, Any]] = {}
    for model_dir in model_dirs:
        eval_path = artifacts_dir / model_dir / "eval.json"
        if not eval_path.exists():
            print(f"Warning: {eval_path} not found, skipping")
            continue

        with open(eval_path, "r", encoding="utf-8") as f:
            eval_results[model_dir] = json.load(f)

    if not eval_results:
        print("Error: No evaluation results found")
        return 1

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate summary.json
    generate_eval_summary(eval_results, output_dir / "summary.json")

    # Generate metrics_bar.png
    plot_metrics_bar(eval_results, output_dir / "metrics_bar.png")

    # Build output string
    output_parts = []
    output_parts.append("=" * 90)
    output_parts.append("MODEL COMPARISON SUMMARY".center(90))
    output_parts.append("=" * 90)

    # Sample info summary
    output_parts.append("\n[SAMPLE INFORMATION]")
    for model_name, data in eval_results.items():
        n_samples = data.get("n_samples", "N/A")
        split_info = data.get("split", {})
        test_size = split_info.get("test_size", "N/A")
        seed = split_info.get("seed", "N/A")
        output_parts.append(f"  {model_name}:")
        output_parts.append(f"    - Total test samples: {n_samples}")
        output_parts.append(f"    - Test split ratio: {test_size}")
        output_parts.append(f"    - Random seed: {seed}")

    # Metrics table
    output_parts.append("\n[PERFORMANCE METRICS]")
    output_parts.append("-" * 90)

    # Header
    header = f"{'Metric':<25}"
    for model_name in eval_results.keys():
        header += f"{model_name:<25}"
    output_parts.append(header)
    output_parts.append("-" * 90)

    # Metrics to display
    metrics = [
        ("Accuracy", "accuracy", "{:.4f}"),
        ("F1 Score", "f1", "{:.4f}"),
        ("ROC-AUC", "roc_auc", "{:.4f}"),
    ]

    for metric_name, metric_key, fmt in metrics:
        row = f"{metric_name:<25}"
        for model_name, data in eval_results.items():
            value = data.get(metric_key)
            if value is None:
                row += f"{'N/A':<25}"
            else:
                row += f"{fmt.format(value):<25}"
        output_parts.append(row)

    # Performance metrics
    output_parts.append("\n[PERFORMANCE & EFFICIENCY]")
    output_parts.append("-" * 90)

    perf_metrics = [
        ("Latency (ms)", "latency", "avg_ms", "{:.4f}"),
        ("Throughput (samples/s)", "throughput", "samples_per_sec", "{:.2f}"),
    ]

    for display_name, section, key, fmt in perf_metrics:
        row = f"{display_name:<25}"
        for model_name, data in eval_results.items():
            section_data = data.get(section, {})
            value = section_data.get(key) if isinstance(section_data, dict) else None
            if value is None:
                row += f"{'N/A':<25}"
            else:
                row += f"{fmt.format(value):<25}"
        output_parts.append(row)

    # Model info
    output_parts.append("\n[MODEL INFORMATION]")
    output_parts.append("-" * 90)
    for model_name, data in eval_results.items():
        output_parts.append(f"  {model_name}:")
        model_type = data.get("model", data.get("artifact_path", "N/A").split("/")[-1].replace(".joblib", ""))
        output_parts.append(f"    - Model type: {model_type}")

        artifact_size = data.get("artifact_size", {})
        if artifact_size:
            output_parts.append(f"    - Artifact size: {artifact_size.get('mb', 'N/A'):.2f} MB")

        preprocess = data.get("preprocess")
        if preprocess:
            preprocess_str = ", ".join(f"{k}={v}" for k, v in preprocess.items())
            output_parts.append(f"    - Preprocessing: {preprocess_str}")

        if data.get("warning"):
            output_parts.append(f"    - Warning: {data.get('warning')}")

    output_parts.append("\n" + "=" * 90)
    output_parts.append(f"Best Model by Accuracy: {max(eval_results.items(), key=lambda x: x[1].get('accuracy', 0))[0]}")
    output_parts.append(f"Best Model by F1: {max(eval_results.items(), key=lambda x: x[1].get('f1', 0))[0]}")
    output_parts.append("=" * 90)

    output_text = "\n".join(output_parts)

    # Print to console
    print(output_text)

    # Write to file
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_txt is None:
        output_txt = output_dir / "model_comparison.txt"

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"\nComparison saved to: {output_txt}")
    return 0


def _build_eval_plot_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eval-plot", description="Generate evaluation plots from model results")
    p.add_argument(
        "--eval-json",
        action="append",
        required=True,
        help="Path to evaluation JSON file (can be specified multiple times)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work") / "eval",
        help="Output directory for plots and summary",
    )
    p.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate metrics bar chart",
    )
    p.add_argument(
        "--summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate JSON summary",
    )
    p.add_argument(
        "--text/--no-text",
        dest="text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print text table to console (default: enabled)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Main entry point supporting both eda and eval-plot commands."""
    parser = argparse.ArgumentParser(prog="run_eda", description="EDA and Evaluation Plotting Tools")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # EDA subcommand
    eda_parser = sub.add_parser("eda", help="Run exploratory data analysis")
    eda_parser.add_argument("--data-path", type=Path, default=Path("work") / "IMDB Dataset.csv")
    eda_parser.add_argument("--output-dir", type=Path, default=Path("work") / "eda")
    eda_parser.add_argument("--strip-html", action=argparse.BooleanOptionalAction, default=True)
    eda_parser.add_argument("--lowercase", action=argparse.BooleanOptionalAction, default=True)
    eda_parser.add_argument("--skip-ngrams", action=argparse.BooleanOptionalAction, default=False)
    eda_parser.add_argument("--top-ngrams", type=int, default=25)
    eda_parser.add_argument("--max-features", type=int, default=50_000)
    eda_parser.add_argument("--min-df", type=int, default=5)
    eda_parser.add_argument("--ngram-sample-size", type=int, default=10_000)
    eda_parser.add_argument("--wordcloud-mode", type=str, default="diff", choices=["by_class", "diff"])
    eda_parser.add_argument("--wordcloud-max-words", type=int, default=100)

    # Eval-plot subcommand
    eval_plot_parser = sub.add_parser("eval-plot", help="Generate evaluation plots from model results")
    eval_plot_parser.add_argument(
        "--eval-json",
        action="append",
        required=True,
        help="Path to evaluation JSON file (can be specified multiple times)",
    )
    eval_plot_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work") / "eval",
        help="Output directory for plots and summary",
    )
    eval_plot_parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate metrics bar chart",
    )
    eval_plot_parser.add_argument(
        "--summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate JSON summary",
    )

    # Compare subcommand
    compare_parser = sub.add_parser("compare", help="Compare evaluation results across all models")
    compare_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("work") / "artifacts",
        help="Directory containing model artifact folders",
    )
    compare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work") / "eda",
        help="Output directory for the comparison text file",
    )
    compare_parser.add_argument(
        "--output-txt",
        type=Path,
        default=None,
        help="Output text file path (default: output-dir/model_comparison.txt)",
    )

    args = parser.parse_args(argv)

    if args.subcommand == "eda":
        # Build argv for eda main
        eda_argv = []
        if args.data_path:
            eda_argv.extend(["--data-path", str(args.data_path)])
        if args.output_dir:
            eda_argv.extend(["--output-dir", str(args.output_dir)])
        if args.strip_html is not None:
            eda_argv.append("--strip-html" if args.strip_html else "--no-strip-html")
        if args.lowercase is not None:
            eda_argv.append("--lowercase" if args.lowercase else "--no-lowercase")
        if args.skip_ngrams:
            eda_argv.append("--skip-ngrams")
        eda_argv.extend(["--top-ngrams", str(args.top_ngrams)])
        eda_argv.extend(["--max-features", str(args.max_features)])
        eda_argv.extend(["--min-df", str(args.min_df)])
        eda_argv.extend(["--ngram-sample-size", str(args.ngram_sample_size)])
        eda_argv.extend(["--wordcloud-mode", str(args.wordcloud_mode)])
        eda_argv.extend(["--wordcloud-max-words", str(args.wordcloud_max_words)])
        return eda_main(eda_argv)

    elif args.subcommand == "eval-plot":
        # Extract only the eval-plot subcommand arguments (exclude subcommand name)
        eval_plot_idx = argv.index("eval-plot") if "eval-plot" in argv else 0
        eval_plot_argv = argv[eval_plot_idx + 1:]
        return eval_plot_main(eval_plot_argv)

    elif args.subcommand == "compare":
        return compare_models_main(args.artifacts_dir, args.output_dir, args.output_txt)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
