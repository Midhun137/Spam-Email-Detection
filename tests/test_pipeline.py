"""Test suite for Spam Email Detection using Ensemble Learning."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from joblib import load

from src.dataset import (
    clean_and_deduplicate,
    compute_content_hash,
    compute_near_duplicate_cluster,
    create_synthetic_fixture,
    extract_sender_domain,
    parse_raw_email,
)
from src.evaluate import compute_metrics
from src.features import (
    EmailFeaturePreparer,
    build_classification_pipeline,
    build_feature_preprocessor,
)
from src.models import get_dummy_classifier, get_voting_ensemble


def test_extract_sender_domain():
    addr, dom, missing = extract_sender_domain("Alice Smith <alice@company.com>")
    assert dom == "company.com"
    assert missing == 0

    addr, dom, missing = extract_sender_domain("invalid-no-at-sign")
    assert dom == "<missing>"
    assert missing == 1

    addr, dom, missing = extract_sender_domain("")
    assert dom == "<missing>"
    assert missing == 1


def test_parse_raw_email_plain():
    raw_email = (
        b"From: bob@example.org\r\n"
        b"Subject: Meeting tomorrow\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n"
        b"\r\n"
        b"Let's meet at 10 AM to discuss the quarterly project deliverables."
    )
    rec = parse_raw_email(raw_email, label=0)
    assert rec is not None
    assert rec["subject"] == "Meeting tomorrow"
    assert rec["sender_domain"] == "example.org"
    assert "quarterly project" in rec["body"]
    assert rec["has_html"] == 0
    assert rec["attachment_count"] == 0
    assert rec["is_sender_missing"] == 0
    assert rec["label"] == 0


def test_parse_raw_email_html_fallback():
    raw_html_email = (
        b"From: promos@shop-online.biz\r\n"
        b"Subject: 50% Off Everything Today Only!\r\n"
        b"Content-Type: text/html; charset=utf-8\r\n"
        b"\r\n"
        b"<html><body><h1>Mega Sale</h1><p>Click <a href='http://shop.biz/sale'>here</a> for discounts!</p></body></html>"
    )
    rec = parse_raw_email(raw_html_email, label=1)
    assert rec is not None
    assert rec["has_html"] == 1
    assert "Mega Sale" in rec["body"]
    assert "Click here for discounts!" in rec["body"]
    assert rec["url_count"] == 1
    assert rec["label"] == 1


def test_missing_sender_and_subject_handling():
    raw_minimal = b"\r\n\r\nJust a body text without any headers."
    rec = parse_raw_email(raw_minimal, label=0)
    assert rec is not None
    assert rec["is_subject_missing"] == 1
    assert rec["is_sender_missing"] == 1
    assert rec["sender_domain"] == "<missing>"
    assert rec["body"] == "Just a body text without any headers."


def test_deduplication_and_conflict_handling():
    records = [
        {"subject": "Hello", "body": "World", "label": 0, "is_subject_missing": 0, "is_body_missing": 0, "is_sender_missing": 0},
        {"subject": "Hello", "body": "World", "label": 0, "is_subject_missing": 0, "is_body_missing": 0, "is_sender_missing": 0},
        {"subject": "Different", "body": "Unique", "label": 1, "is_subject_missing": 0, "is_body_missing": 0, "is_sender_missing": 0},
    ]
    df = pd.DataFrame(records)
    cleaned_df, stats = clean_and_deduplicate(df)
    assert len(cleaned_df) == 2
    assert stats["exact_duplicates_removed"] == 1
    assert stats["conflicting_duplicates_removed"] == 0


def test_feature_preprocessor_non_negative():
    data = [
        {
            "subject": "Discount luxury watches",
            "sender": "deals@watches.xyz",
            "body": "Buy Rolex replicas at http://replica.xyz! Huge savings! Call 1800555 now!",
        },
        {
            "subject": "Code review update",
            "sender": "colleague@work.org",
            "body": "PR 42 has been merged into main.",
        },
    ]
    preparer = EmailFeaturePreparer()
    prepared_df = preparer.fit_transform(data)

    preprocessor = build_feature_preprocessor(feature_set="all")
    X_sparse = preprocessor.fit_transform(prepared_df)

    # Check non-negativity (crucial for MultinomialNB)
    assert X_sparse.data.min() >= 0.0
    assert X_sparse.data.max() <= 1.0


def test_false_positive_definition():
    """Verify that a legitimate email (ham=0) marked as spam (1) is counted as a false positive."""
    # y_true = 0 (ham), y_pred = 1 (spam) -> MUST be False Positive
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    # Index 0: TN (0, 0)
    # Index 1: FP (0, 1) - Legitimate marked as spam
    # Index 2: FN (1, 0) - Spam missed
    # Index 3: TP (1, 1) - Spam caught

    m = compute_metrics(y_true, y_pred)
    assert m["true_negatives"] == 1
    assert m["false_positives"] == 1
    assert m["false_negatives"] == 1
    assert m["true_positives"] == 1
    assert m["accuracy"] == 0.5


def test_saved_pipeline_prediction():
    pipeline_path = Path("models/spam_ensemble_pipeline.joblib")
    if not pipeline_path.exists():
        pytest.skip("Trained pipeline not found on disk")

    pipeline = load(pipeline_path)

    # Legitimate email test
    ham_input = pd.DataFrame(
        [
            {
                "subject": "Quarterly financial budget review",
                "sender": "finance@corporation.com",
                "body": "Hi team, attached is the Q3 financial summary. Please review prior to Monday's planning session.",
            }
        ]
    )
    ham_pred = pipeline.predict(ham_input)[0]
    ham_prob = pipeline.predict_proba(ham_input)[0]
    assert ham_pred == 0  # Should be classified as Ham
    assert ham_prob[0] > ham_prob[1]
    assert 0.0 <= ham_prob[1] <= 1.0

    # Spam email test
    spam_input = pd.DataFrame(
        [
            {
                "subject": "URGENT: Claim $1,000,000 lottery cash prize now!!!",
                "sender": "winner@instant-cash-prize.biz",
                "body": "CONGRATULATIONS! You won big money! Click http://cash-claim-free.biz/win now to collect your prize! Call 1-800-555-CASH immediately!",
            }
        ]
    )
    spam_pred = pipeline.predict(spam_input)[0]
    spam_prob = pipeline.predict_proba(spam_input)[0]
    assert spam_pred == 1  # Should be classified as Spam
    assert spam_prob[1] > spam_prob[0]
    assert 0.0 <= spam_prob[1] <= 1.0


def test_synthetic_fixture_generation(tmp_path):
    fixture_csv = create_synthetic_fixture(tmp_path)
    assert fixture_csv.exists()
    df = pd.read_csv(fixture_csv, escapechar="\\")
    assert len(df) == 20
    assert (df["label"] == 0).sum() == 10
    assert (df["label"] == 1).sum() == 10


def test_generated_reports_and_figures_exist():
    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    expected_figures = [
        "class_distribution.png",
        "model_comparison.png",
        "confusion_matrix_ensemble.png",
        "confusion_matrix_best_individual.png",
        "precision_recall_curves.png",
        "roc_curves.png",
        "feature_ablation.png",
    ]
    for fig_name in expected_figures:
        fig_path = figures_dir / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_name}"
        assert fig_path.stat().st_size > 0, f"Empty figure: {fig_name}"

    assert (reports_dir / "metrics.json").exists()
    assert (reports_dir / "metrics.csv").exists()
    assert (reports_dir / "ablation_metrics.json").exists()
    assert (reports_dir / "error_analysis.json").exists()
