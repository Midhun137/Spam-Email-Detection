"""Feature engineering pipelines for email spam classification.

Implements three feature groups:
1. Text: TF-IDF on combined subject and body with word unigrams and bigrams.
2. Sender: One-hot encoded sender domain (with rare/unseen handling) and missing-sender flag.
3. Message characteristics: Lengths, URL counts, attachment counts, uppercase/digit ratios,
   exclamation counts, and HTML indicators, scaled to [0, 1] for non-negativity.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder

URL_PATTERN = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE)

NUMERICAL_MESSAGE_FEATURES = [
    "subject_len",
    "body_len",
    "url_count",
    "attachment_count",
    "digit_ratio",
    "uppercase_ratio",
    "exclamation_count",
    "has_html",
    "is_sender_missing",
]


class EmailFeaturePreparer(BaseEstimator, TransformerMixin):
    """Ensures input DataFrame contains all expected raw and derived feature columns.

    Works identically on raw data records from training files or single
    dictionaries passed during inference/Streamlit demo.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: Any, y: Any = None) -> "EmailFeaturePreparer":
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        """Transforms input dictionaries or DataFrame into a standard feature DataFrame."""
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, (list, tuple)):
            df = pd.DataFrame(list(X))
        elif isinstance(X, dict):
            df = pd.DataFrame([X])
        else:
            df = pd.DataFrame(X)

        # Standardize text columns
        for col in ["subject", "body"]:
            if col not in df.columns:
                df[col] = ""
            else:
                df[col] = df[col].fillna("").astype(str)

        # Build combined text
        df["text"] = "Subject: " + df["subject"] + "\n\n" + df["body"]

        # Handle sender domain
        if "sender_domain" not in df.columns:
            sender_col = None
            for candidate in ["sender_address", "sender", "from", "From"]:
                if candidate in df.columns:
                    sender_col = candidate
                    break

            domains: List[str] = []
            is_missing: List[int] = []
            if sender_col:
                for val in df[sender_col]:
                    val_str = str(val or "").strip()
                    if "@" in val_str:
                        dom = val_str.split("@")[-1].strip().lower()
                        dom = re.sub(r"[^\w.-]", "", dom)
                        if dom and "." in dom:
                            domains.append(dom)
                            is_missing.append(0)
                        else:
                            domains.append("<missing>")
                            is_missing.append(1)
                    else:
                        domains.append("<missing>")
                        is_missing.append(1)
            else:
                domains = ["<missing>"] * len(df)
                is_missing = [1] * len(df)

            df["sender_domain"] = domains
            df["is_sender_missing"] = is_missing
        else:
            df["sender_domain"] = df["sender_domain"].fillna("<missing>").astype(str)
            if "is_sender_missing" not in df.columns:
                df["is_sender_missing"] = (df["sender_domain"] == "<missing>").astype(int)

        # Derive message characteristics if absent
        if "subject_len" not in df.columns:
            df["subject_len"] = df["subject"].str.len()

        if "body_len" not in df.columns:
            df["body_len"] = df["body"].str.len()

        if "url_count" not in df.columns:
            df["url_count"] = df["body"].apply(lambda b: len(URL_PATTERN.findall(str(b))))

        if "attachment_count" not in df.columns:
            df["attachment_count"] = 0

        if "digit_ratio" not in df.columns:
            df["digit_ratio"] = df["body"].apply(
                lambda b: sum(c.isdigit() for c in str(b)) / max(len(str(b)), 1)
            )

        if "uppercase_ratio" not in df.columns:
            df["uppercase_ratio"] = df["body"].apply(
                lambda b: sum(c.isupper() for c in str(b))
                / max(sum(c.isalpha() for c in str(b)), 1)
            )

        if "exclamation_count" not in df.columns:
            df["exclamation_count"] = (df["subject"] + " " + df["body"]).apply(
                lambda t: str(t).count("!")
            )

        if "has_html" not in df.columns:
            df["has_html"] = df["body"].apply(
                lambda b: 1
                if ("<html" in str(b).lower() or "<body" in str(b).lower() or "<p" in str(b).lower())
                else 0
            )

        # Ensure correct numerical types
        for col in NUMERICAL_MESSAGE_FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        return df


def build_text_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """Build TF-IDF vectorizer for subject and body text."""
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,
        stop_words="english",
        lowercase=True,
        strip_accents="unicode",
    )


def build_sender_encoder(min_frequency: int = 5) -> OneHotEncoder:
    """Build one-hot encoder for sender domains."""
    return OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=min_frequency,
        sparse_output=True,
    )


def build_message_feature_transformer() -> Pipeline:
    """Build transformer for message characteristics.

    Applies log1p for heavily skewed count features and MinMax scaling to [0, 1].
    The [0, 1] range ensures strict non-negativity for MultinomialNB compatibility.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("log1p", FunctionTransformer(np.log1p, validate=False)),
            ("scaler", MinMaxScaler(feature_range=(0.0, 1.0))),
        ]
    )


def build_feature_preprocessor(
    feature_set: str = "all",
    max_text_features: int = 5000,
    min_sender_frequency: int = 5,
) -> ColumnTransformer:
    """Build ColumnTransformer for selected feature group ablation.

    Supported options for feature_set:
    - 'text_only': Text TF-IDF features only.
    - 'text_sender': Text TF-IDF + Sender domain features.
    - 'all' / 'text_sender_message': Text + Sender + Message structural features.
    """
    transformers = []

    # 1. Text feature pipeline
    text_tf = build_text_vectorizer(max_features=max_text_features)
    transformers.append(("text", text_tf, "text"))

    if feature_set in ("text_sender", "all", "text_sender_message"):
        # 2. Sender domain pipeline
        sender_ohe = build_sender_encoder(min_frequency=min_sender_frequency)
        transformers.append(("sender", sender_ohe, ["sender_domain"]))

    if feature_set in ("all", "text_sender_message"):
        # 3. Message characteristics pipeline
        msg_tf = build_message_feature_transformer()
        transformers.append(("message", msg_tf, NUMERICAL_MESSAGE_FEATURES))

    return ColumnTransformer(
        transformers=transformers,
        sparse_threshold=1.0,  # Maintain CSR sparse matrix
        remainder="drop",
    )


def build_classification_pipeline(
    classifier: Any,
    feature_set: str = "all",
    max_text_features: int = 5000,
) -> Pipeline:
    """Construct an end-to-end pipeline ready for raw data fitting and prediction."""
    return Pipeline(
        [
            ("preparer", EmailFeaturePreparer()),
            (
                "preprocessor",
                build_feature_preprocessor(
                    feature_set=feature_set,
                    max_text_features=max_text_features,
                ),
            ),
            ("classifier", classifier),
        ]
    )
