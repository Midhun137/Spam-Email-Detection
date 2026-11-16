"""Generate presentation-ready figures for individual models and slide deck."""

import json
import os
import tempfile
from pathlib import Path

os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

with open("reports/metrics.json", "r") as f:
    data = json.load(f)

results = data["test_results"]

MODELS_CONFIG = [
    {
        "key": "Multinomial Naive Bayes",
        "display_name": "Multinomial Naive Bayes",
        "file_name": "cm_multinomial_nb.png",
        "cmap": "Blues",
        "subtitle": "Generative model with Laplace smoothing (alpha=0.1)",
    },
    {
        "key": "Logistic Regression",
        "display_name": "Logistic Regression",
        "file_name": "cm_logistic_regression.png",
        "cmap": "Greens",
        "subtitle": "L2-regularized linear model with Sigmoid activation",
    },
    {
        "key": "Calibrated Linear SVM",
        "display_name": "Calibrated Linear Support Vector Classifier",
        "file_name": "cm_calibrated_svm.png",
        "cmap": "Purples",
        "subtitle": "Maximum-margin hyperplane with Platt Sigmoid calibration",
    },
    {
        "key": "Ensemble (Soft - Tuned Weights)",
        "display_name": "Soft-Voting Ensemble (Tuned Weights: 1.0, 0.25, 1.5)",
        "file_name": "cm_soft_ensemble.png",
        "cmap": "Oranges",
        "subtitle": "Fused posterior probabilities of NB, LR, and Calibrated SVM",
    },
]

# 1. Generate individual PPT slide figures (Confusion matrix + formatted scorecard)
for cfg in MODELS_CONFIG:
    m = results[cfg["key"]]
    cm = np.array(m["confusion_matrix"])

    fig = plt.figure(figsize=(10, 5.2), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85], wspace=0.25)

    # Left: Confusion Matrix Heatmap
    ax0 = fig.add_subplot(gs[0])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cfg["cmap"],
        cbar=False,
        xticklabels=["Pred Ham", "Pred Spam"],
        yticklabels=["True Ham", "True Spam"],
        annot_kws={"size": 18, "weight": "bold"},
        ax=ax0,
    )
    ax0.set_title(
        f"{cfg['display_name']}\n{cfg['subtitle']}",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax0.set_xlabel("Predicted Label (0 = Ham, 1 = Spam)", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Actual Ground Truth Label", fontsize=11, fontweight="bold")

    # Right: Metric Summary Card
    ax1 = fig.add_subplot(gs[1])
    ax1.axis("off")

    lines = [
        "PERFORMANCE METRICS",
        "=" * 28,
        f"Accuracy        : {m['accuracy']*100:.2f}%",
        f"Spam Precision  : {m['precision']*100:.2f}%",
        f"Spam Recall     : {m['recall']*100:.2f}%",
        f"Spam F1 Score   : {m['f1']:.4f}",
        f"ROC-AUC         : {m['roc_auc']:.4f}",
        f"Average Prec.   : {m['average_precision']:.4f}",
        "-" * 28,
        f"True Negatives  : {m['true_negatives']} (Ham caught)",
        f"False Positives : {m['false_positives']} (Ham in Spam)",
        f"False Negatives : {m['false_negatives']} (Spam missed)",
        f"True Positives  : {m['true_positives']} (Spam caught)",
    ]
    card_content = "\n".join(lines)

    ax1.text(
        0.05,
        0.5,
        card_content,
        transform=ax1.transAxes,
        fontsize=11.5,
        family="monospace",
        verticalalignment="center",
        bbox=dict(
            boxstyle="round,pad=0.9",
            facecolor="#f8f9fa",
            edgecolor="#adb5bd",
            linewidth=1.5,
        ),
    )

    out_file = FIGURES_DIR / cfg["file_name"]
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_file}")

# 2. Combined 4-Panel Side-by-Side Grid for 16:9 Slide
fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), dpi=300)
for ax, cfg in zip(axes.flat, MODELS_CONFIG):
    m = results[cfg["key"]]
    cm = np.array(m["confusion_matrix"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cfg["cmap"],
        cbar=False,
        xticklabels=["Pred Ham", "Pred Spam"],
        yticklabels=["True Ham", "True Spam"],
        annot_kws={"size": 15, "weight": "bold"},
        ax=ax,
    )
    f1_val = m["f1"]
    prec_val = m["precision"] * 100
    rec_val = m["recall"] * 100
    ax.set_title(
        f"{cfg['display_name']}\nF1: {f1_val:.4f}  |  Precision: {prec_val:.1f}%  |  Recall: {rec_val:.1f}%",
        fontsize=11.5,
        fontweight="bold",
        pad=8,
    )
    ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=10, fontweight="bold")

fig.suptitle(
    "Comparative Confusion Matrices on Held-Out Test Set (N = 901 Emails)",
    fontsize=16,
    fontweight="bold",
    y=0.99,
)
fig.tight_layout()
combined_out = FIGURES_DIR / "all_models_confusion_matrices.png"
fig.savefig(combined_out, bbox_inches="tight")
plt.close(fig)
print(f"Generated: {combined_out}")

# 3. High-Contrast PPT Slide Bar Chart for Metrics Comparison
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
names = [cfg["display_name"].split("(")[0].strip() for cfg in MODELS_CONFIG]
f1_scores = [results[cfg["key"]]["f1"] for cfg in MODELS_CONFIG]
prec_scores = [results[cfg["key"]]["precision"] for cfg in MODELS_CONFIG]
rec_scores = [results[cfg["key"]]["recall"] for cfg in MODELS_CONFIG]

x = np.arange(len(names))
width = 0.25

rects1 = ax.bar(x - width, prec_scores, width, label="Spam Precision", color="#2b5c8f", edgecolor="black")
rects2 = ax.bar(x, rec_scores, width, label="Spam Recall", color="#388e3c", edgecolor="black")
rects3 = ax.bar(x + width, f1_scores, width, label="Spam F1 (Primary)", color="#d32f2f", edgecolor="black")

ax.set_title("Model Performance Comparison on Held-Out Test Set", fontsize=14, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=11, fontweight="bold")
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0.92, 1.0)
ax.legend(loc="lower right", frameon=True, fontsize=11)

# Annotate F1 on the bars
for bar in rects3:
    height = bar.get_height()
    ax.annotate(
        f"{height:.4f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )

fig.tight_layout()
ppt_bar_out = FIGURES_DIR / "model_comparison_presentation.png"
fig.savefig(ppt_bar_out, bbox_inches="tight")
plt.close(fig)
print(f"Generated: {ppt_bar_out}")
