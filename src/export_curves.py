"""Export exact test-set precision-recall and ROC curves for instant Plotly dashboard rendering."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.evaluate import create_group_aware_split

DATA_PATH = Path("data/processed/emails_cleaned.csv")
MODEL_PATH = Path("models/spam_ensemble_pipeline.joblib")
OUT_PATH = Path("reports/curves_data.json")


def export_curves() -> None:
    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        print("Data or model file not found.")
        return

    df = pd.read_csv(DATA_PATH, escapechar="\\")
    train_df, test_df, _ = create_group_aware_split(df, test_size=0.20)
    pipeline = load(MODEL_PATH)

    preparer = pipeline.named_steps["preparer"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    prep_test = preparer.transform(test_df)
    X_test = preprocessor.transform(prep_test)
    y_test = test_df["label"].values

    curves_data = {}

    # 1. Ensemble
    p_ens = classifier.predict_proba(X_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, p_ens, pos_label=1)
    fpr, tpr, _ = roc_curve(y_test, p_ens, pos_label=1)

    # Subsample curves to ~100 points for lightweight instant Plotly rendering
    def downsample(x, y, n=120):
        if len(x) <= n:
            return [round(float(v), 4) for v in x], [round(float(v), 4) for v in y]
        idx = np.linspace(0, len(x) - 1, n).astype(int)
        return [round(float(x[i]), 4) for i in idx], [round(float(y[i]), 4) for i in idx]

    p_d, r_d = downsample(prec, rec)
    fpr_d, tpr_d = downsample(fpr, tpr)

    curves_data["Ensemble (Soft - Tuned)"] = {
        "precision": p_d,
        "recall": r_d,
        "fpr": fpr_d,
        "tpr": tpr_d,
        "ap": round(float(average_precision_score(y_test, p_ens)), 4),
        "auc": round(float(roc_auc_score(y_test, p_ens)), 4),
    }

    # 2. Sub-estimators
    display_names = {
        "nb": "Multinomial Naive Bayes",
        "lr": "Logistic Regression",
        "svm": "Calibrated Linear SVM",
    }

    for name, clf in classifier.named_estimators_.items():
        if name in display_names:
            p = clf.predict_proba(X_test)[:, 1]
            prec, rec, _ = precision_recall_curve(y_test, p, pos_label=1)
            fpr, tpr, _ = roc_curve(y_test, p, pos_label=1)
            p_d, r_d = downsample(prec, rec)
            fpr_d, tpr_d = downsample(fpr, tpr)
            curves_data[display_names[name]] = {
                "precision": p_d,
                "recall": r_d,
                "fpr": fpr_d,
                "tpr": tpr_d,
                "ap": round(float(average_precision_score(y_test, p)), 4),
                "auc": round(float(roc_auc_score(y_test, p)), 4),
            }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(curves_data, f, indent=2)

    print(f"Saved curve data to {OUT_PATH} with {len(curves_data)} models.")


if __name__ == "__main__":
    export_curves()
