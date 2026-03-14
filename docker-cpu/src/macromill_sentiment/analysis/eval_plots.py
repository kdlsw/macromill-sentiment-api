"""
Evaluation visualization utilities.

This module provides functions to visualize model evaluation metrics.
"""
from __future__ import annotations

import json
from pathlib import Path


def plot_metrics_bar(eval_results: dict, output_path: Path) -> None:
    """
    Create a bar chart comparing metrics across multiple models.

    Args:
        eval_results: Dictionary mapping model names to their evaluation results.
                     Each result should contain keys like 'accuracy', 'f1', 'roc_auc'.
        output_path: Path to save the output PNG image.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not eval_results:
        raise ValueError("eval_results cannot be empty")

    # Collect all metric names from the first model
    model_names = list(eval_results.keys())
    first_result = eval_results[model_names[0]]
    
    # Identify numeric metrics (exclude non-metric fields)
    exclude_keys = {"artifact_path", "preprocess", "n_samples", "split", "latency", "throughput", "artifact_size", "model", "warning"}
    metric_names = [k for k in first_result.keys() if k not in exclude_keys and isinstance(first_result[k], (int, float))]

    if not metric_names:
        raise ValueError("No numeric metrics found in evaluation results")

    # Build data matrix
    data = {}
    for metric in metric_names:
        values = []
        for model_name in model_names:
            val = eval_results[model_name].get(metric)
            if val is None:
                val = 0.0  # or could use NaN
            values.append(val)
        data[metric] = values

    # Create bar chart
    x = np.arange(len(model_names))
    width = 0.8 / len(metric_names)
    colors = plt.cm.Set2(np.linspace(0, 1, len(metric_names)))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (metric, values) in enumerate(data.items()):
        offset = (i - len(metric_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric.replace("_", " ").title(), color=colors[i])

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved metrics bar chart to {output_path}")


def plot_confusion_matrix(
    confusion_matrix: list[list[int]],
    labels: list[str] | None = None,
    output_path: Path | None = None,
    normalize: bool = False,
) -> None:
    """
    Create a heatmap visualization of a confusion matrix.

    Args:
        confusion_matrix: 2x2 confusion matrix as nested list [[TN, FP], [FN, TP]].
        labels: Label names for the axes. Defaults to ["negative", "positive"].
        output_path: Path to save the output PNG image. If None, returns the figure.
        normalize: If True, convert counts to proportions.

    Note:
        This function is currently a placeholder.
        To implement, use sklearn's confusion_matrix display or matplotlib with seaborn.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.array(confusion_matrix)
    
    if labels is None:
        labels = ["negative", "positive"]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Placeholder implementation - creates a simple heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels,
           yticklabels=labels,
           title='Confusion Matrix' + (' (Normalized)' if normalize else ''),
           ylabel='True label',
           xlabel='Predicted label')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved confusion matrix to {output_path}")
    else:
        return fig


def plot_roc_curve(
    y_true: list[int],
    y_scores: list[float],
    output_path: Path | None = None,
    model_name: str = "Model",
) -> dict | None:
    """
    Plot ROC curve and calculate AUC.

    Args:
        y_true: True binary labels (0 or 1).
        y_scores: Predicted probabilities or decision scores.
        output_path: Path to save the output PNG image. If None, returns the figure.
        model_name: Name of the model for the plot title.

    Returns:
        Dictionary with 'fpr', 'tpr', 'auc' keys, or None if plotting only.

    Note:
        This function is currently a placeholder.
        To implement fully, use sklearn.metrics.roc_curve and sklearn.metrics.roc_auc_score.
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt
    import numpy as np

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved ROC curve to {output_path}")
        return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc_score}
    else:
        return fig


def generate_eval_summary(eval_results: dict, output_path: Path) -> None:
    """
    Generate a comprehensive evaluation summary with metrics table.

    Args:
        eval_results: Dictionary mapping model names to their evaluation results.
        output_path: Path to save the summary JSON file.
    """
    summary: dict = {
        "models": {},
        "comparison": {},
    }

    # Collect all metrics
    all_metrics = set()
    for model_name, results in eval_results.items():
        metric_keys = [k for k in results.keys() 
                      if isinstance(results[k], (int, float)) and k not in {"n_samples"}]
        all_metrics.update(metric_keys)

    # Build per-model summary
    for model_name, results in eval_results.items():
        model_summary: dict = {}
        
        # Core metrics
        for metric in ["accuracy", "f1", "roc_auc"]:
            if metric in results:
                model_summary[metric] = results[metric]
        
        # Confusion matrix
        if "confusion_matrix" in results:
            cm = results["confusion_matrix"]
            if cm and len(cm) == 2:
                model_summary["confusion_matrix"] = {
                    "TN": cm[0][0],
                    "FP": cm[0][1],
                    "FN": cm[1][0],
                    "TP": cm[1][1],
                }
        
        # Performance metrics
        if "latency" in results:
            model_summary["latency_ms"] = results["latency"]["avg_ms"]
        
        if "throughput" in results:
            model_summary["throughput_samples_per_sec"] = results["throughput"]["samples_per_sec"]
        
        if "artifact_size" in results:
            model_summary["artifact_size_mb"] = results["artifact_size"]["mb"]
        
        summary["models"][model_name] = model_summary

    # Add comparison section - find best for each metric
    for metric in ["accuracy", "f1", "roc_auc"]:
        best_model = None
        best_value = -1
        for model_name, results in eval_results.items():
            val = results.get(metric)
            if val is not None and val > best_value:
                best_value = val
                best_model = model_name
        
        if best_model:
            summary["comparison"][metric] = {
                "best_model": best_model,
                "value": best_value,
            }

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved evaluation summary to {output_path}")


def print_metrics_table(eval_results: dict) -> str:
    """Generate a text table comparing metrics across multiple models.
    
    Args:
        eval_results: Dictionary mapping model names to their evaluation results.
    
    Returns:
        Formatted string with the comparison table.
    """
    if not eval_results:
        return "No evaluation results to display."
    
    model_names = list(eval_results.keys())
    first_result = eval_results[model_names[0]]
    
    # Identify metric rows (excluding non-metric fields)
    exclude_keys = {
        "artifact_path", "preprocess", "n_samples", "split", "model", "warning",
        "batch_size", "eval_batch_size", "total_params", "trainable_params"
    }
    
    # Build list of metrics to display
    metric_rows = []
    
    # Core metrics (in order)
    core_metrics = ["accuracy", "f1", "roc_auc"]
    for m in core_metrics:
        if m in first_result:
            metric_rows.append(m)
    
    # Confusion matrix (special handling)
    if "confusion_matrix" in first_result:
        metric_rows.append("confusion_matrix")
    
    # Performance metrics - check both flat keys and nested structures
    perf_metrics = [
        ("latency", "latency_ms", "avg_ms"),
        ("throughput", "throughput_samples_per_sec", "samples_per_sec"),
        ("artifact_size", "artifact_size_mb", "mb"),
    ]
    for nested_key, flat_key, sub_key in perf_metrics:
        if flat_key in first_result:
            metric_rows.append(flat_key)
        elif nested_key in first_result and isinstance(first_result[nested_key], dict):
            metric_rows.append(flat_key)
    
    # Add any other numeric metrics not yet listed
    for key, val in first_result.items():
        if key not in exclude_keys and key not in metric_rows and isinstance(val, (int, float)):
            metric_rows.append(key)
    
    # Calculate column widths
    row_label_width = max(len(r) for r in metric_rows) + 2
    col_widths = [max(len(m) for m in model_names)] + [12] * len(model_names)
    col_widths[0] = max(col_widths[0], row_label_width)
    
    # Build header
    header = " " * col_widths[0] + "  " + "  ".join(
        m.ljust(w) for m, w in zip(model_names, col_widths[1:])
    )
    separator = "-" * len(header)
    
    # Build rows
    lines = [header, separator]
    
    for row in metric_rows:
        if row == "confusion_matrix":
            # Confusion matrix rows
            cm_keys = ["TN", "FP", "FN", "TP"]
            for cm_key in cm_keys:
                row_label = f"  {cm_key}"
                cells = []
                for model_name in model_names:
                    cm = eval_results[model_name].get("confusion_matrix")
                    if cm:
                        # Handle both list format [[TN, FP], [FN, TP]] and dict format
                        if isinstance(cm, list) and len(cm) == 2:
                            cm_dict = {"TN": cm[0][0], "FP": cm[0][1], "FN": cm[1][0], "TP": cm[1][1]}
                            val = cm_dict.get(cm_key, 0)
                            cells.append(f"{float(val):12.2f}")
                        elif isinstance(cm, dict):
                            val = cm.get(cm_key, 0)
                            cells.append(f"{float(val):12.2f}")
                        else:
                            cells.append(f"{'N/A':>12}")
                    else:
                        cells.append(f"{'N/A':>12}")
                lines.append(row_label.ljust(col_widths[0]) + "  " + "  ".join(cells))
        else:
            row_label = row
            cells = []
            for model_name in model_names:
                # Check for flat key first, then nested
                val = eval_results[model_name].get(row)
                if val is None:
                    # Try nested lookup for performance metrics
                    for nested_key, flat_key, sub_key in perf_metrics:
                        if row == flat_key and nested_key in eval_results[model_name]:
                            nested = eval_results[model_name][nested_key]
                            if isinstance(nested, dict) and sub_key in nested:
                                val = nested[sub_key]
                                break
                if val is not None:
                    if isinstance(val, float):
                        cells.append(f"{val:12.4f}")
                    else:
                        cells.append(f"{str(val):>12}")
                else:
                    cells.append(f"{'N/A':>12}")
            lines.append(row_label.ljust(col_widths[0]) + "  " + "  ".join(cells))
    
    lines.append(separator)
    
    # Add best models by metric
    best_lines = ["Best Models by Metric:"]
    for metric in core_metrics:
        if metric not in first_result:
            continue
        best_model = None
        best_value = -1
        for model_name in model_names:
            val = eval_results[model_name].get(metric)
            if val is not None and val > best_value:
                best_value = val
                best_model = model_name
        if best_model:
            best_lines.append(f"  {metric:12} -> {best_model} ({best_value:.4f})")
    
    if best_lines:
        lines.append("")
        lines.extend(best_lines)
    
    return "\n".join(lines)
