"""Investigate term presence in vocabulary and training corpus."""

import json
from pathlib import Path
import pandas as pd
from joblib import load

df = pd.read_csv("data/processed/emails_cleaned.csv", escapechar="\\")
pipeline = load("models/spam_ensemble_pipeline.joblib")
preprocessor = pipeline.named_steps["preprocessor"]
vec = preprocessor.named_transformers_["text"]
vocab = vec.vocabulary_
classifier = pipeline.named_steps["classifier"]
lr = classifier.named_estimators_["lr"]
nb = classifier.named_estimators_["nb"]

terms = [
    "account", "suspended", "verify", "identity", "banking", "services",
    "gift", "apple", "wire", "transfer", "watches", "luxury", "discount", "free"
]

print("=" * 65)
print(f"{'Term':<12} | {'In Vocab':<8} | {'Ham Cnt':<8} | {'Spam Cnt':<8} | {'LR Coeff':<9}")
print("=" * 65)
for term in terms:
    in_vocab = term in vocab
    idx = vocab.get(term, -1)
    ham_has = int(df[df["label"] == 0]["body"].str.contains(term, case=False, na=False).sum())
    spam_has = int(df[df["label"] == 1]["body"].str.contains(term, case=False, na=False).sum())
    lr_coeff = round(float(lr.coef_[0, idx]), 4) if idx >= 0 else None
    coeff_str = str(lr_coeff) if lr_coeff is not None else "N/A"
    print(f"{term:<12} | {str(in_vocab):<8} | {ham_has:<8} | {spam_has:<8} | {coeff_str:<9}")
print("=" * 65)
