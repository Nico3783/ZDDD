"""Dissertation-grade visualization module for model evaluation.

Generates publication-quality plots for Chapters 3 & 4 including:
ROC curves, confusion matrices, per-class metrics, model comparison,
feature importance, anomaly score distributions, and threshold analysis.

All plots are saved as high-resolution PNG files to reports/figures/.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Matplotlib configuration for publication quality
_MPL_CONFIGURED = False


def _configure_matplotlib() -> None:
    """Configure matplotlib for publication-quality output."""
    global _MPL_CONFIGURED
    if _MPL_CONFIGURED:
        return

    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    _MPL_CONFIGURED = True


# Color palette (professional, colorblind-safe)
COLORS = {
    "primary": "#1976D2",
    "secondary": "#388E3C",
    "accent": "#F57C00",
    "danger": "#D32F2F",
    "purple": "#7B1FA2",
    "teal": "#00796B",
    "class": [
        "#1976D2",  # Blue
        "#D32F2F",  # Red
        "#388E3C",  # Green
        "#F57C00",  # Orange
        "#7B1FA2",  # Purple
        "#00796B",  # Teal
    ],
}


def _ensure_output_dir(output_dir: str | Path) -> Path:
    """Create output directory if it doesn't exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_roc_curve_binary(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "ROC Curve — Isolation Forest",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot binary ROC curve with AUC annotation.

    Args:
        y_true: Ground truth binary labels (0/1).
        y_scores: Predicted anomaly scores.
        title: Plot title.
        save_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt
    from src.evaluation.roc import compute_roc_curve, compute_roc_auc

    _configure_matplotlib()

    fpr, tpr, _ = compute_roc_curve(y_true.astype(int), y_scores)
    auc = compute_roc_auc(y_true.astype(int), y_scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color=COLORS["primary"], lw=2.5,
            label=f"Isolation Forest (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--", alpha=0.6, label="Random Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        logger.info("ROC curve saved to %s", save_path)
    plt.close(fig)


def plot_roc_curve_multiclass(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_names: List[str],
    title: str = "ROC Curve — Random Forest (One-vs-Rest)",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot multiclass ROC curves with per-class AUC.

    Args:
        y_true: Ground truth labels (integer-encoded).
        y_scores: Predicted probabilities (n_samples x n_classes).
        class_names: List of class names.
        title: Plot title.
        save_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt
    from src.evaluation.roc import compute_roc_curve, compute_roc_auc

    _configure_matplotlib()

    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(9, 7))

    for i in range(n_classes):
        binary_true = (y_true == i).astype(int)
        class_scores = y_scores[:, i]
        fpr, tpr, _ = compute_roc_curve(binary_true, class_scores)
        auc = compute_roc_auc(binary_true, class_scores)
        color = COLORS["class"][i % len(COLORS["class"])]
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{class_names[i]} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--", alpha=0.6, label="Random Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        logger.info("Multiclass ROC curve saved to %s", save_path)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str | Path] = None,
    normalize: bool = True,
) -> None:
    """Plot confusion matrix as a annotated heatmap.

    Args:
        y_true: Ground truth labels (integer-encoded).
        y_pred: Predicted labels (integer-encoded).
        class_names: List of class names.
        title: Plot title.
        save_path: Path to save the figure.
        normalize: If True, show percentages instead of counts.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    _configure_matplotlib()

    n_classes = len(class_names)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            matrix[t, p] += 1

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        display = matrix.astype(float) / row_sums
        fmt = ".2%"
        vmin, vmax = 0.0, 1.0
    else:
        display = matrix.astype(float)
        fmt = "d"
        vmin, vmax = 0, matrix.max()

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(display, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")

    # Annotate cells
    thresh = (vmax + vmin) / 2
    for i in range(n_classes):
        for j in range(n_classes):
            val = display[i, j]
            text_val = f"{val:{fmt}}" if normalize else f"{int(val):,}"
            color = "white" if val > thresh else "black"
            ax.text(j, i, text_val, ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion" if normalize else "Count", fontsize=10)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        logger.info("Confusion matrix saved to %s", save_path)
    plt.close(fig)


def plot_per_class_metrics(
    per_class: Dict[str, Dict[str, float]],
    title: str = "Per-Class Classification Metrics",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot grouped bar chart of per-class precision, recall, F1.

    Args:
        per_class: Dict mapping class names to metric dicts.
        title: Plot title.
        save_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt

    _configure_matplotlib()

    classes = sorted(per_class.keys())
    precisions = [per_class[c]["precision"] for c in classes]
    recalls = [per_class[c]["recall"] for c in classes]
    f1s = [per_class[c]["f1_score"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precisions, width, label="Precision", color=COLORS["primary"], alpha=0.9)
    bars2 = ax.bar(x, recalls, width, label="Recall", color=COLORS["secondary"], alpha=0.9)
    bars3 = ax.bar(x + width, f1s, width, label="F1-Score", color=COLORS["accent"], alpha=0.9)

    ax.set_xlabel("Attack Class")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right", framealpha=0.9)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        logger.info("Per-class metrics saved to %s", save_path)
    plt.close(fig)


def plot_model_comparison(
    iforest_metrics: Dict[str, float],
    rf_metrics: Dict[str, float],
    title: str = "Model Performance Comparison",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot side-by-side comparison of Isolation Forest vs Random Forest.

    Args:
        iforest_metrics: Metric dict for Isolation Forest.
        rf_metrics: Metric dict for Random Forest.
        title: Plot title.
        save_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt

    _configure_matplotlib()

    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    display_names = ["Accuracy", "Precision", "Recall", "F1-Score"]

    if_values = [iforest_metrics.get(m, 0) for m in metric_names]
    rf_values = [rf_metrics.get(m, 0) for m in metric_names]

    x = np.arange(len(display_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, if_values, width, label="Isolation Forest",
                   color=COLORS["primary"], alpha=0.9)
    bars2 = ax.bar(x + width / 2, rf_values, width, label="Random Forest",
                   color=COLORS["secondary"], alpha=0.9)

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.set_ylim([0.0, 1.1])
    ax.legend(loc="upper left", framealpha=0.9)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.4f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        logger.info("Model comparison saved to %s", save_path)
    plt.close(fig)


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_k: int = 20,
    title: str = "Feature Importance — Random Forest",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot horizontal bar chart of top feature importances.

    Args:
        feature_names: List of feature names.
        importances: Array of importance scores.
        top_k: Number of top features to display.
        title: Plot title.
        save_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt

    _configure_matplotlib()

    # Sort and take top_k
    indices = np.argsort(importances)[::-1][:top_k]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_importances, color=COLORS["primary"], alpha=0.9, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    ax.invert_yaxis()

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, top_importances)):
        ax.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=8)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        logger.info("Feature importance saved to %s", save_path)
    plt.close(fig)


def plot_anomaly_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    title: str = "Anomaly Score Distribution — Isolation Forest",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot histogram of anomaly scores colored by true class.

    Args:
        scores: Normalized anomaly scores (0-1).
        labels: True binary labels (0=normal, 1=anomaly).
        threshold: Detection threshold line.
        title: Plot title.
        save_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt

    _configure_matplotlib()

    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(normal_scores, bins=50, alpha=0.7, color=COLORS["secondary"],
            label=f"Benign (n={len(normal_scores):,})", density=True, edgecolor="white")
    ax.hist(anomaly_scores, bins=50, alpha=0.7, color=COLORS["danger"],
            label=f"Attack (n={len(anomaly_scores):,})", density=True, edgecolor="white")
    ax.axvline(threshold, color=COLORS["accent"], lw=2, ls="--",
               label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        logger.info("Anomaly score distribution saved to %s", save_path)
    plt.close(fig)


def plot_threshold_analysis(
    y_true: np.ndarray,
    scores: np.ndarray,
    title: str = "Threshold Sensitivity Analysis — Isolation Forest",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot precision, recall, F1 vs threshold for anomaly detection.

    Args:
        y_true: Ground truth binary labels.
        scores: Normalized anomaly scores.
        title: Plot title.
        save_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt

    _configure_matplotlib()

    thresholds = np.linspace(0.0, 1.0, 200)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    best_idx = int(np.argmax(f1s))
    best_threshold = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precisions, color=COLORS["primary"], lw=2, label="Precision")
    ax.plot(thresholds, recalls, color=COLORS["secondary"], lw=2, label="Recall")
    ax.plot(thresholds, f1s, color=COLORS["accent"], lw=2, label="F1-Score")
    ax.axvline(best_threshold, color=COLORS["danger"], lw=1.5, ls="--",
               label=f"Best F1 Threshold = {best_threshold:.4f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="center left", framealpha=0.9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        logger.info("Threshold analysis saved to %s", save_path)
    plt.close(fig)


def plot_classification_report(
    per_class: Dict[str, Dict[str, float]],
    overall: Dict[str, float],
    title: str = "Classification Report — Random Forest",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot classification report as a styled table image.

    Args:
        per_class: Per-class metric dict.
        overall: Overall metric dict (accuracy, etc.).
        title: Plot title.
        save_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    _configure_matplotlib()

    classes = sorted(per_class.keys())
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    cell_data = []
    for cls in classes:
        m = per_class[cls]
        cell_data.append([
            cls,
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1_score']:.4f}",
            f"{m['support']:,}",
        ])

    # Add overall row
    cell_data.append([
        "Weighted Avg",
        f"{overall.get('precision', 0):.4f}",
        f"{overall.get('recall', 0):.4f}",
        f"{overall.get('f1_score', 0):.4f}",
        f"{sum(per_class[c]['support'] for c in classes):,}",
    ])

    fig, ax = plt.subplots(figsize=(10, 0.5 * (len(classes) + 2) + 1.5))
    ax.axis("off")
    ax.set_title(title, pad=20, fontweight="bold")

    table = ax.table(
        cellText=cell_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Header styling
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor(COLORS["primary"])
        cell.set_text_props(color="white", fontweight="bold")

    # Row styling
    for i in range(1, len(cell_data) + 1):
        for j in range(len(headers)):
            cell = table[i, j]
            if i == len(cell_data):  # Overall row
                cell.set_facecolor("#E3F2FD")
                cell.set_text_props(fontweight="bold")
            elif i % 2 == 0:
                cell.set_facecolor("#F5F5F5")

    # Add accuracy annotation
    accuracy = overall.get("accuracy", 0)
    ax.text(0.5, -0.02, f"Overall Accuracy: {accuracy:.4f}",
            ha="center", va="top", transform=ax.transAxes, fontsize=12,
            fontweight="bold", color=COLORS["primary"])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        logger.info("Classification report saved to %s", save_path)
    plt.close(fig)


def generate_all_visualizations(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    if_scores: np.ndarray,
    if_threshold: float,
    y_true_multiclass: np.ndarray,
    y_pred_multiclass: np.ndarray,
    rf_probabilities: np.ndarray,
    class_names: List[str],
    feature_names: List[str],
    feature_importances: np.ndarray,
    if_metrics: Dict[str, float],
    rf_metrics: Dict[str, float],
    if_per_class: Dict[str, Dict[str, float]],
    rf_per_class: Dict[str, Dict[str, float]],
    output_dir: str | Path = "reports/figures",
) -> List[str]:
    """Generate all dissertation-grade visualizations.

    Args:
        y_true_binary: True binary labels for IF evaluation.
        y_pred_binary: Predicted binary labels from IF.
        if_scores: Normalized anomaly scores from IF.
        if_threshold: IF detection threshold.
        y_true_multiclass: True multi-class labels (int-encoded) for RF.
        y_pred_multiclass: Predicted multi-class labels (int-encoded) from RF.
        rf_probabilities: RF predicted probabilities.
        class_names: List of class names.
        feature_names: Feature names used by RF.
        feature_importances: RF feature importances.
        if_metrics: IF overall metrics dict.
        rf_metrics: RF overall metrics dict.
        if_per_class: IF per-class metrics.
        rf_per_class: RF per-class metrics.
        output_dir: Directory to save all figures.

    Returns:
        List of saved file paths.
    """
    out = _ensure_output_dir(output_dir)
    saved: List[str] = []

    logger.info("Generating dissertation visualizations to %s", out)

    # 1. ROC Curve — Isolation Forest (Binary)
    path = out / "roc_curve_iforest.png"
    plot_roc_curve_binary(
        y_true_binary, if_scores,
        title="ROC Curve — Isolation Forest Anomaly Detector",
        save_path=path,
    )
    saved.append(str(path))

    # 2. ROC Curves — Random Forest (Multiclass)
    path = out / "roc_curve_random_forest.png"
    plot_roc_curve_multiclass(
        y_true_multiclass, rf_probabilities, class_names,
        title="ROC Curves — Random Forest Multi-Class Classifier",
        save_path=path,
    )
    saved.append(str(path))

    # 3. Confusion Matrix — Isolation Forest
    path = out / "confusion_matrix_iforest.png"
    plot_confusion_matrix(
        y_true_binary, y_pred_binary,
        class_names=["Benign", "Attack"],
        title="Confusion Matrix — Isolation Forest",
        save_path=path,
        normalize=True,
    )
    saved.append(str(path))

    # 4. Confusion Matrix — Random Forest
    path = out / "confusion_matrix_random_forest.png"
    plot_confusion_matrix(
        y_true_multiclass, y_pred_multiclass, class_names,
        title="Confusion Matrix — Random Forest",
        save_path=path,
        normalize=True,
    )
    saved.append(str(path))

    # 5. Per-Class Metrics — Random Forest
    path = out / "per_class_metrics_rf.png"
    plot_per_class_metrics(
        rf_per_class,
        title="Per-Class Precision, Recall, and F1-Score — Random Forest",
        save_path=path,
    )
    saved.append(str(path))

    # 6. Model Comparison
    path = out / "model_comparison.png"
    plot_model_comparison(
        if_metrics, rf_metrics,
        title="Model Performance Comparison — Isolation Forest vs Random Forest",
        save_path=path,
    )
    saved.append(str(path))

    # 7. Feature Importance — Random Forest
    path = out / "feature_importance_rf.png"
    plot_feature_importance(
        feature_names, feature_importances, top_k=20,
        title="Top 20 Feature Importances — Random Forest",
        save_path=path,
    )
    saved.append(str(path))

    # 8. Anomaly Score Distribution — Isolation Forest
    path = out / "anomaly_score_distribution.png"
    plot_anomaly_score_distribution(
        if_scores, y_true_binary, threshold=if_threshold,
        title="Anomaly Score Distribution — Isolation Forest",
        save_path=path,
    )
    saved.append(str(path))

    # 9. Threshold Sensitivity Analysis — Isolation Forest
    path = out / "threshold_analysis_iforest.png"
    plot_threshold_analysis(
        y_true_binary, if_scores,
        title="Threshold Sensitivity Analysis — Isolation Forest",
        save_path=path,
    )
    saved.append(str(path))

    # 10. Classification Report Table — Random Forest
    path = out / "classification_report_rf.png"
    overall_rf = {
        "accuracy": rf_metrics.get("accuracy", 0),
        "precision": rf_metrics.get("precision", 0),
        "recall": rf_metrics.get("recall", 0),
        "f1_score": rf_metrics.get("f1_score", 0),
    }
    plot_classification_report(
        rf_per_class, overall_rf,
        title="Classification Report — Random Forest",
        save_path=path,
    )
    saved.append(str(path))

    logger.info("All %d visualizations generated successfully", len(saved))
    return saved
