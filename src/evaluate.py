"""Evaluation pipeline, cross-validation, feature ablation, and visualization.

Executes:
- Group-aware stratified train/test split (preventing near-duplicate leakage)
- 5-fold cross-validation on the training set for tuning and weight optimization
- Test set evaluation for all individual models and soft-voting ensemble
- Feature ablation comparison (Text vs Text+Sender vs All)
- Error analysis on false positives and false negatives
- Export of metric tables (CSV, JSON) and publication-quality figures (PNG)
- Serialization of the final production pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure headless matplotlib with writable config
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold

from src.features import (
    EmailFeaturePreparer,
    build_classification_pipeline,
    build_feature_preprocessor,
)
from src.models import (
    DEFAULT_RANDOM_STATE,
    get_calibrated_svm,
    get_dummy_classifier,
    get_logistic_regression,
    get_naive_bayes,
    get_voting_ensemble,
    optimize_soft_voting_weights,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_group_aware_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Partition dataset into train and test sets respecting near-duplicate groups."""
    sgkf = StratifiedGroupKFold(
        n_splits=int(round(1.0 / test_size)),
        shuffle=True,
        random_state=random_state,
    )
    splits = list(sgkf.split(df, df["label"], groups=df["duplicate_group_id"]))
    test_fold_idx = 0
    train_indices, test_indices = splits[test_fold_idx]

    train_df = df.iloc[train_indices].copy().reset_index(drop=True)
    test_df = df.iloc[test_indices].copy().reset_index(drop=True)

    # Validate group separation
    train_groups = set(train_df["duplicate_group_id"])
    test_groups = set(test_df["duplicate_group_id"])
    overlap = train_groups.intersection(test_groups)

    stats = {
        "train_size": len(train_df),
        "train_ham_count": int((train_df["label"] == 0).sum()),
        "train_spam_count": int((train_df["label"] == 1).sum()),
        "train_spam_pct": round(float(train_df["label"].mean() * 100), 2),
        "test_size": len(test_df),
        "test_ham_count": int((test_df["label"] == 0).sum()),
        "test_spam_count": int((test_df["label"] == 1).sum()),
        "test_spam_pct": round(float(test_df["label"].mean() * 100), 2),
        "group_overlap_count": len(overlap),
    }

    return train_df, test_df, stats


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics for positive class (Spam = 1)."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    metrics: Dict[str, Any] = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm.tolist(),
    }

    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            avg_prec = average_precision_score(y_true, y_prob)
            metrics["roc_auc"] = round(float(roc_auc), 4)
            metrics["average_precision"] = round(float(avg_prec), 4)
        except Exception:
            metrics["roc_auc"] = None
            metrics["average_precision"] = None
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    return metrics


def run_cross_validation(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[Dict[str, Any], List[float]]:
    """Perform 5-fold cross-validation on the training set to optimize ensemble weights."""
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    oof_nb_probs = np.zeros((len(train_df), 2))
    oof_lr_probs = np.zeros((len(train_df), 2))
    oof_svm_probs = np.zeros((len(train_df), 2))

    for fold, (trn_idx, val_idx) in enumerate(
        sgkf.split(train_df, train_df["label"], groups=train_df["duplicate_group_id"])
    ):
        fold_train = train_df.iloc[trn_idx]
        fold_val = train_df.iloc[val_idx]

        # Fit feature pipeline strictly inside fold
        preparer = EmailFeaturePreparer()
        prep_train = preparer.fit_transform(fold_train)
        prep_val = preparer.transform(fold_val)

        ct = build_feature_preprocessor(feature_set="all")
        X_trn = ct.fit_transform(prep_train)
        X_val = ct.transform(prep_val)
        y_trn = fold_train["label"].values

        # 1. NB
        nb = get_naive_bayes()
        nb.fit(X_trn, y_trn)
        oof_nb_probs[val_idx] = nb.predict_proba(X_val)

        # 2. LR
        lr = get_logistic_regression()
        lr.fit(X_trn, y_trn)
        oof_lr_probs[val_idx] = lr.predict_proba(X_val)

        # 3. SVM
        svm = get_calibrated_svm()
        svm.fit(X_trn, y_trn)
        oof_svm_probs[val_idx] = svm.predict_proba(X_val)

    y_train = train_df["label"].values
    best_weights, best_cv_f1 = optimize_soft_voting_weights(
        oof_nb_probs, oof_lr_probs, oof_svm_probs, y_train
    )

    # Evaluate equal weights CV F1
    p_equal = (oof_nb_probs[:, 1] + oof_lr_probs[:, 1] + oof_svm_probs[:, 1]) / 3.0
    equal_f1 = f1_score(y_train, (p_equal >= 0.5).astype(int), pos_label=1)

    cv_results = {
        "equal_weights": [1.0, 1.0, 1.0],
        "equal_weights_cv_f1": round(float(equal_f1), 4),
        "tuned_weights": best_weights,
        "tuned_weights_cv_f1": round(float(best_cv_f1), 4),
    }

    logger.info(
        f"Training CV Results -> Equal Weights F1: {equal_f1:.4f} | Tuned Weights {best_weights} F1: {best_cv_f1:.4f}"
    )

    return cv_results, best_weights


def evaluate_models_on_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tuned_weights: List[float],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, np.ndarray]]:
    """Train estimators on full training partition and evaluate on untouched test set."""
    models: Dict[str, Any] = {
        "Dummy (Majority)": get_dummy_classifier(),
        "Multinomial Naive Bayes": get_naive_bayes(),
        "Logistic Regression": get_logistic_regression(),
        "Calibrated Linear SVM": get_calibrated_svm(),
        "Ensemble (Soft - Equal Weights)": get_voting_ensemble(weights=[1.0, 1.0, 1.0]),
        f"Ensemble (Soft - Tuned Weights)": get_voting_ensemble(weights=tuned_weights),
    }

    results: Dict[str, Dict[str, Any]] = {}
    test_probs: Dict[str, np.ndarray] = {}

    X_train = train_df
    y_train = train_df["label"].values
    X_test = test_df
    y_test = test_df["label"].values

    for name, clf in models.items():
        logger.info(f"Training pipeline for {name}...")
        pipeline = build_classification_pipeline(clf, feature_set="all")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X_test)[:, 1]
            test_probs[name] = probs
        else:
            probs = None

        metrics = compute_metrics(y_test, y_pred, y_prob=probs)
        results[name] = metrics
        logger.info(
            f"{name} -> Accuracy: {metrics['accuracy']:.4f}, Spam Precision: {metrics['precision']:.4f}, "
            f"Spam Recall: {metrics['recall']:.4f}, Spam F1: {metrics['f1']:.4f}"
        )

    return results, test_probs


def run_feature_ablation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    weights: List[float],
) -> Dict[str, Dict[str, Any]]:
    """Evaluate performance across feature sets: Text Only, Text+Sender, and All."""
    ablation_sets = [
        ("Text Only", "text_only"),
        ("Text + Sender", "text_sender"),
        ("Text + Sender + Message Characteristics", "all"),
    ]

    ablation_results: Dict[str, Dict[str, Any]] = {}
    y_test = test_df["label"].values

    for label, feature_mode in ablation_sets:
        logger.info(f"Running feature ablation: {label}...")
        clf = get_voting_ensemble(weights=weights)
        pipeline = build_classification_pipeline(clf, feature_set=feature_mode)
        pipeline.fit(train_df, train_df["label"].values)

        y_pred = pipeline.predict(test_df)
        probs = pipeline.predict_proba(test_df)[:, 1]

        m = compute_metrics(y_test, y_pred, y_prob=probs)
        ablation_results[label] = m

        logger.info(
            f"Ablation [{label}] -> F1: {m['f1']:.4f}, Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}"
        )

    return ablation_results


def perform_error_analysis(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    max_examples: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Inspect false positives and false negatives without exposing sensitive details."""
    y_test = test_df["label"].values

    fps_mask = (y_test == 0) & (y_pred == 1)
    fns_mask = (y_test == 1) & (y_pred == 0)

    fp_indices = np.where(fps_mask)[0]
    fn_indices = np.where(fns_mask)[0]

    false_positives = []
    for idx in fp_indices[:max_examples]:
        row = test_df.iloc[idx]
        subj = str(row["subject"])[:60] if pd.notna(row.get("subject")) and str(row.get("subject")).strip() else "<no subject>"
        false_positives.append(
            {
                "index": int(idx),
                "true_label": "Ham (0)",
                "predicted_label": "Spam (1)",
                "estimated_spam_prob": round(float(y_prob[idx]), 4),
                "subject_summary": subj,
                "sender_domain": str(row.get("sender_domain", "<missing>")),
                "has_html": int(row.get("has_html", 0)),
                "url_count": int(row.get("url_count", 0)),
                "exclamation_count": int(row.get("exclamation_count", 0)),
                "reason_pattern": "Legitimate email with promotional/newsletter tone, HTML formatting, or high URL/link frequency.",
            }
        )

    false_negatives = []
    for idx in fn_indices[:max_examples]:
        row = test_df.iloc[idx]
        subj = str(row["subject"])[:60] if pd.notna(row.get("subject")) and str(row.get("subject")).strip() else "<no subject>"
        false_negatives.append(
            {
                "index": int(idx),
                "true_label": "Spam (1)",
                "predicted_label": "Ham (0)",
                "estimated_spam_prob": round(float(y_prob[idx]), 4),
                "subject_summary": subj,
                "sender_domain": str(row.get("sender_domain", "<missing>")),
                "has_html": int(row.get("has_html", 0)),
                "url_count": int(row.get("url_count", 0)),
                "exclamation_count": int(row.get("exclamation_count", 0)),
                "reason_pattern": "Short or obfuscated spam message lacking common spam trigger vocabulary or masquerading as discussion.",
            }
        )

    return {
        "total_false_positives": int(len(fp_indices)),
        "total_false_negatives": int(len(fn_indices)),
        "sample_false_positives": false_positives,
        "sample_false_negatives": false_negatives,
    }


def generate_plots(
    df_cleaned: pd.DataFrame,
    results: Dict[str, Dict[str, Any]],
    ablation_results: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    test_probs: Dict[str, np.ndarray],
    best_indiv_name: str,
    ensemble_name: str,
    output_dir: Path,
) -> None:
    """Generate high-resolution PNG figures for academic report and presentation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Cleaned class distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df_cleaned["label"].value_counts().sort_index()
    labels = ["Ham (Legitimate)", "Spam (Unsolicited)"]
    palette = ["#2b5c8f", "#d95f02"]
    bars = ax.bar(labels, counts.values, color=palette, width=0.55, edgecolor="black", linewidth=1.2)
    for bar in bars:
        height = bar.get_height()
        pct = 100.0 * height / len(df_cleaned)
        ax.annotate(
            f"{height:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height / 2),
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=12,
        )
    ax.set_title("Cleaned Dataset Class Distribution (SpamAssassin Corpus)", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Emails", fontsize=12)
    ax.set_ylim(0, max(counts.values) * 1.15)
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=300)
    plt.close(fig)

    # 2. Model comparison grouped bar chart
    fig, ax = plt.subplots(figsize=(11, 6))
    model_names = [m for m in results.keys() if m != "Dummy (Majority)"]
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Spam Precision", "Spam Recall", "Spam F1 (Primary)"]

    x = np.arange(len(model_names))
    width = 0.2
    palette_comp = ["#4e79a7", "#59a14f", "#edc949", "#e15759"]

    for i, (m_key, m_lbl) in enumerate(zip(metrics_to_plot, metric_labels)):
        vals = [results[m][m_key] for m in model_names]
        rects = ax.bar(x + i * width, vals, width, label=m_lbl, color=palette_comp[i], edgecolor="black", linewidth=0.8)

    ax.set_title("Model Performance Comparison on Held-Out Test Set", fontsize=15, fontweight="bold", pad=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.replace("Ensemble ", "Ens. ") for m in model_names], rotation=15, ha="right", fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0.85, 1.01)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=300)
    plt.close(fig)

    # 3. Confusion Matrix - Ensemble
    fig, ax = plt.subplots(figsize=(6, 5))
    ens_cm = np.array(results[ensemble_name]["confusion_matrix"])
    sns.heatmap(
        ens_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred Ham", "Pred Spam"],
        yticklabels=["True Ham", "True Spam"],
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(f"Confusion Matrix: {ensemble_name}", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_ensemble.png", dpi=300)
    plt.close(fig)

    # 4. Confusion Matrix - Best Individual Classifier
    fig, ax = plt.subplots(figsize=(6, 5))
    indiv_cm = np.array(results[best_indiv_name]["confusion_matrix"])
    sns.heatmap(
        indiv_cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        cbar=False,
        xticklabels=["Pred Ham", "Pred Spam"],
        yticklabels=["True Ham", "True Spam"],
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(f"Confusion Matrix: {best_indiv_name}", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_best_individual.png", dpi=300)
    plt.close(fig)

    # 5. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probs in test_probs.items():
        if name == "Dummy (Majority)":
            continue
        p, r, _ = precision_recall_curve(y_test, probs, pos_label=1)
        ap = results[name]["average_precision"]
        ax.plot(r, p, lw=2.2, label=f"{name} (AP = {ap:.4f})")

    ax.set_title("Precision-Recall Curves (Spam = Positive Class)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Recall (Spam Caught)", fontsize=12)
    ax.set_ylabel("Precision (True Spam / Flagged)", fontsize=12)
    ax.set_xlim([0.80, 1.01])
    ax.set_ylim([0.80, 1.01])
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curves.png", dpi=300)
    plt.close(fig)

    # 6. ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probs in test_probs.items():
        if name == "Dummy (Majority)":
            continue
        fpr, tpr, _ = roc_curve(y_test, probs, pos_label=1)
        auc_val = results[name]["roc_auc"]
        ax.plot(fpr, tpr, lw=2.2, label=f"{name} (AUC = {auc_val:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC = 0.50)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (Legitimate Flagged)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Spam Caught)", fontsize=12)
    ax.set_xlim([-0.01, 0.20])
    ax.set_ylim([0.80, 1.01])
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=300)
    plt.close(fig)

    # 7. Feature Ablation Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ab_labels = list(ablation_results.keys())
    ab_f1s = [ablation_results[k]["f1"] for k in ab_labels]
    ab_precs = [ablation_results[k]["precision"] for k in ab_labels]
    ab_recs = [ablation_results[k]["recall"] for k in ab_labels]

    x = np.arange(len(ab_labels))
    w = 0.26
    ax.bar(x - w, ab_precs, w, label="Precision", color="#4e79a7", edgecolor="black")
    ax.bar(x, ab_recs, w, label="Recall", color="#59a14f", edgecolor="black")
    ax.bar(x + w, ab_f1s, w, label="Spam F1 (Primary)", color="#e15759", edgecolor="black")

    ax.set_title("Feature Group Ablation Performance (Soft-Voting Ensemble)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(ab_labels, fontsize=11)
    ax.set_ylabel("Metric Score", fontsize=12)
    ax.set_ylim(0.90, 1.01)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_ablation.png", dpi=300)
    plt.close(fig)

    logger.info(f"Saved all 7 publication figures to {output_dir}")


def train_and_evaluate(
    data_path: Path,
    output_dir: Path,
    model_dir: Path,
) -> Dict[str, Any]:
    """Execute complete end-to-end training, cross-validation, evaluation, and serialization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading cleaned dataset from {data_path}...")
    df = pd.read_csv(data_path, escapechar="\\")
    logger.info(f"Loaded {len(df)} emails (Ham: {(df['label']==0).sum()}, Spam: {(df['label']==1).sum()})")

    # 1. Group-aware train/test partition
    train_df, test_df, split_stats = create_group_aware_split(df, test_size=0.20)
    logger.info(
        f"Partitioned: Train={split_stats['train_size']} ({split_stats['train_spam_pct']}% spam), "
        f"Test={split_stats['test_size']} ({split_stats['test_spam_pct']}% spam). Group overlap: {split_stats['group_overlap_count']}"
    )

    # 2. Cross-validation on training data to optimize ensemble weights
    logger.info("Executing 5-fold cross-validation on training partition...")
    cv_results, best_weights = run_cross_validation(train_df, n_splits=5)

    # 3. Evaluate models on held-out test set
    logger.info("Evaluating all candidate models on held-out test partition...")
    test_results, test_probs = evaluate_models_on_test(train_df, test_df, best_weights)

    # Identify best individual classifier by Spam F1
    individual_models = [
        "Multinomial Naive Bayes",
        "Logistic Regression",
        "Calibrated Linear SVM",
    ]
    best_indiv_name = max(individual_models, key=lambda m: test_results[m]["f1"])
    ensemble_key = "Ensemble (Soft - Tuned Weights)"
    if ensemble_key not in test_results:
        ensemble_key = "Ensemble (Soft - Equal Weights)"

    logger.info(f"Strongest Individual Classifier: {best_indiv_name} (F1: {test_results[best_indiv_name]['f1']:.4f})")
    logger.info(f"Ensemble Performance: {ensemble_key} (F1: {test_results[ensemble_key]['f1']:.4f})")

    # 4. Feature ablation study
    ablation_results = run_feature_ablation(train_df, test_df, best_weights)

    # 5. Error analysis on test predictions
    best_ensemble_probs = test_probs[ensemble_key]
    best_ensemble_preds = (best_ensemble_probs >= 0.5).astype(int)
    error_analysis = perform_error_analysis(test_df, best_ensemble_preds, best_ensemble_probs)

    # 6. Generate presentation-ready PNG figures
    generate_plots(
        df_cleaned=df,
        results=test_results,
        ablation_results=ablation_results,
        y_test=test_df["label"].values,
        test_probs=test_probs,
        best_indiv_name=best_indiv_name,
        ensemble_name=ensemble_key,
        output_dir=figures_dir,
    )

    # 7. Serialize final production pipeline trained on all data (or full train split)
    logger.info("Training and saving final production pipeline...")
    final_pipeline = build_classification_pipeline(
        get_voting_ensemble(weights=best_weights),
        feature_set="all",
    )
    final_pipeline.fit(train_df, train_df["label"].values)

    pipeline_path = model_dir / "spam_ensemble_pipeline.joblib"
    dump(final_pipeline, pipeline_path)
    logger.info(f"Saved production model pipeline to {pipeline_path}")

    # Metadata and parameters
    metadata = {
        "model_type": "SoftVotingClassifier",
        "sub_estimators": ["MultinomialNB", "LogisticRegression", "CalibratedClassifierCV(LinearSVC)"],
        "voting": "soft",
        "weights": best_weights,
        "feature_groups": ["text_tfidf_1_2_ngrams", "sender_domain_ohe", "message_characteristics_scaled"],
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "strongest_individual_classifier": best_indiv_name,
        "best_individual_f1": test_results[best_indiv_name]["f1"],
        "ensemble_f1": test_results[ensemble_key]["f1"],
        "random_state": DEFAULT_RANDOM_STATE,
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 8. Export structured report files (CSV and JSON)
    metrics_summary = {
        "split_stats": split_stats,
        "cross_validation_tuning": cv_results,
        "test_results": test_results,
        "best_individual_classifier": best_indiv_name,
        "feature_ablation": ablation_results,
        "error_analysis_summary": {
            "total_false_positives": error_analysis["total_false_positives"],
            "total_false_negatives": error_analysis["total_false_negatives"],
        },
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # CSV report
    metrics_rows = []
    for model_name, m in test_results.items():
        metrics_rows.append(
            {
                "Model": model_name,
                "Accuracy": m["accuracy"],
                "Spam Precision": m["precision"],
                "Spam Recall": m["recall"],
                "Spam F1": m["f1"],
                "ROC-AUC": m["roc_auc"],
                "Average Precision": m["average_precision"],
                "True Negatives": m["true_negatives"],
                "False Positives": m["false_positives"],
                "False Negatives": m["false_negatives"],
                "True Positives": m["true_positives"],
            }
        )
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(output_dir / "metrics.csv", index=False)
    logger.info(f"Saved metrics CSV to {output_dir / 'metrics.csv'}")

    with open(output_dir / "ablation_metrics.json", "w") as f:
        json.dump(ablation_results, f, indent=2)

    with open(output_dir / "error_analysis.json", "w") as f:
        json.dump(error_analysis, f, indent=2)

    return metrics_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate spam email ensemble models")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/emails_cleaned.csv",
        help="Path to cleaned tabular dataset CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save evaluation metrics, tables, and figures",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory to save the serialized pipeline and metadata",
    )
    args = parser.parse_args()

    metrics = train_and_evaluate(
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
        model_dir=Path(args.model_dir),
    )

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION RESULTS (Spam = Positive Class)")
    print("=" * 60)
    for model_name, m in metrics["test_results"].items():
        print(
            f"{model_name:<34} | Acc: {m['accuracy']:.4f} | Prec: {m['precision']:.4f} | "
            f"Rec: {m['recall']:.4f} | F1: {m['f1']:.4f} | AUC: {str(m['roc_auc']):<6}"
        )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
