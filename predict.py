"""Command-line prediction script for Spam Email Detection using the trained ensemble."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from joblib import load

DEFAULT_MODEL_PATH = Path("models/spam_ensemble_pipeline.joblib")


def predict_email(
    subject: str = "",
    sender: str = "",
    body: str = "",
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict:
    """Predict spam probability and class label for a single email."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at '{model_path}'. "
            "Please run 'python -m src.evaluate' first to train and generate the model artifact."
        )

    pipeline = load(model_path)

    # Wrap input in single-row DataFrame
    input_df = pd.DataFrame(
        [
            {
                "subject": subject or "",
                "sender": sender or "",
                "sender_address": sender or "",
                "body": body or "",
            }
        ]
    )

    pred_class = int(pipeline.predict(input_df)[0])
    spam_prob = float(pipeline.predict_proba(input_df)[0][1])

    label_str = "Spam" if pred_class == 1 else "Ham"

    return {
        "prediction": label_str,
        "is_spam": bool(pred_class == 1),
        "spam_probability": round(spam_prob, 4),
        "ham_probability": round(1.0 - spam_prob, 4),
        "subject": subject,
        "sender": sender,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict spam/ham for an email message")
    parser.add_argument("--subject", type=str, default="", help="Email subject line")
    parser.add_argument("--sender", type=str, default="", help="Sender email address (From header)")
    parser.add_argument("--body", type=str, default="", help="Email body text")
    parser.add_argument("--file", type=str, default=None, help="Path to text or raw email file")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to saved joblib pipeline",
    )
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    subject = args.subject
    sender = args.sender
    body = args.body

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            file_content = f.read()
        if not body:
            body = file_content

    try:
        result = predict_email(
            subject=subject,
            sender=sender,
            body=body,
            model_path=Path(args.model_path),
        )
    except Exception as e:
        print(f"Prediction Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 45)
        print("EMAIL CLASSIFICATION RESULT")
        print("=" * 45)
        print(f"Prediction       : {result['prediction'].upper()}")
        print(f"Spam Probability : {result['spam_probability'] * 100:.2f}%")
        print(f"Ham Probability  : {result['ham_probability'] * 100:.2f}%")
        print(f"Subject          : {result['subject'] or '<empty>'}")
        print(f"Sender           : {result['sender'] or '<missing>'}")
        print("=" * 45 + "\n")


if __name__ == "__main__":
    main()
