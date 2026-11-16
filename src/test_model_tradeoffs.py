"""Systematic validation comparison of alternative voting weights and decision thresholds."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from joblib import load
from src.evaluate import create_group_aware_split, run_cross_validation
from src.features import EmailFeaturePreparer, build_feature_preprocessor
from src.models import get_naive_bayes, get_logistic_regression, get_calibrated_svm
from sklearn.model_selection import StratifiedGroupKFold

DATA_PATH = Path("data/processed/emails_cleaned.csv")
df = pd.read_csv(DATA_PATH, escapechar="\\")
train_df, test_df, split_stats = create_group_aware_split(df, test_size=0.20)

print(f"Loaded {len(train_df)} training emails ({int((train_df['label']==1).sum())} spam, {int((train_df['label']==0).sum())} ham).")

# Run 5-fold CV on training data to collect out-of-fold probabilities
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_nb = np.zeros((len(train_df), 2))
oof_lr = np.zeros((len(train_df), 2))
oof_svm = np.zeros((len(train_df), 2))

for fold, (trn_idx, val_idx) in enumerate(sgkf.split(train_df, train_df["label"], groups=train_df["duplicate_group_id"])):
    fold_train = train_df.iloc[trn_idx]
    fold_val = train_df.iloc[val_idx]
    
    preparer = EmailFeaturePreparer()
    prep_trn = preparer.fit_transform(fold_train)
    prep_val = preparer.transform(fold_val)
    
    ct = build_feature_preprocessor(feature_set="all")
    X_trn = ct.fit_transform(prep_trn)
    X_val = ct.transform(prep_val)
    y_trn = fold_train["label"].values
    
    nb = get_naive_bayes()
    nb.fit(X_trn, y_trn)
    oof_nb[val_idx] = nb.predict_proba(X_val)
    
    lr = get_logistic_regression()
    lr.fit(X_trn, y_trn)
    oof_lr[val_idx] = lr.predict_proba(X_val)
    
    svm = get_calibrated_svm()
    svm.fit(X_trn, y_trn)
    oof_svm[val_idx] = svm.predict_proba(X_val)

y_true = train_df["label"].values

candidate_configs = [
    ("Current Tuned [1.0, 0.25, 1.5] @ tau=0.50", [1.0, 0.25, 1.5], 0.50),
    ("Current Tuned [1.0, 0.25, 1.5] @ tau=0.45", [1.0, 0.25, 1.5], 0.45),
    ("Current Tuned [1.0, 0.25, 1.5] @ tau=0.40", [1.0, 0.25, 1.5], 0.40),
    ("Equal Weights [1.0, 1.0, 1.0] @ tau=0.50", [1.0, 1.0, 1.0], 0.50),
    ("Equal Weights [1.0, 1.0, 1.0] @ tau=0.45", [1.0, 1.0, 1.0], 0.45),
    ("Balanced NB-SVM [1.5, 0.5, 1.5] @ tau=0.50", [1.5, 0.5, 1.5], 0.50),
    ("Balanced NB-SVM [1.5, 0.5, 1.5] @ tau=0.45", [1.5, 0.5, 1.5], 0.45),
    ("Higher NB [1.5, 0.25, 1.0] @ tau=0.50", [1.5, 0.25, 1.0], 0.50),
    ("Higher NB [1.5, 0.25, 1.0] @ tau=0.45", [1.5, 0.25, 1.0], 0.45),
]

print("\n" + "=" * 90)
print(f"{'Configuration':<42} | {'Spam Missed (FN)':<16} | {'Ham Flagged (FP)':<16} | {'Prec':<7} | {'Rec':<7} | {'F1':<7}")
print("=" * 90)

for name, w, tau in candidate_configs:
    w_sum = sum(w)
    p_spam = (w[0]*oof_nb[:, 1] + w[1]*oof_lr[:, 1] + w[2]*oof_svm[:, 1]) / w_sum
    preds = (p_spam >= tau).astype(int)
    
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec = precision_score(y_true, preds, pos_label=1, zero_division=0)
    rec = recall_score(y_true, preds, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
    
    print(f"{name:<42} | {fn:<16} | {fp:<16} | {prec*100:.2f}% | {rec*100:.2f}% | {f1:.4f}")

print("=" * 90)
