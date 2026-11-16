"""Model definitions and soft-voting ensemble for spam email detection.

Implements:
1. DummyClassifier (majority-class baseline)
2. MultinomialNB (Naive Bayes on non-negative sparse features)
3. LogisticRegression (L2 regularized)
4. Calibrated LinearSVC (Linear SVM with Platt sigmoid calibration)
5. Soft-Voting Ensemble combining MultinomialNB, LogisticRegression, and LinearSVC
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

DEFAULT_RANDOM_STATE = 42


def get_dummy_classifier(strategy: str = "most_frequent") -> DummyClassifier:
    """Majority-class baseline classifier."""
    return DummyClassifier(strategy=strategy, random_state=DEFAULT_RANDOM_STATE)


def get_naive_bayes(alpha: float = 0.1) -> MultinomialNB:
    """Multinomial Naive Bayes with Laplace/Lidstone additive smoothing.

    Requires non-negative feature matrices.
    """
    return MultinomialNB(alpha=alpha, force_alpha=True)


def get_logistic_regression(
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> LogisticRegression:
    """L2-regularized Logistic Regression with L-BFGS solver."""
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=random_state,
        class_weight=None,
    )


def get_calibrated_svm(
    C: float = 1.0,
    cv: int = 3,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> CalibratedClassifierCV:
    """Linear Support Vector Classifier with Platt sigmoid probability calibration."""
    base_svc = LinearSVC(
        C=C,
        max_iter=2000,
        random_state=random_state,
        dual="auto",
    )
    return CalibratedClassifierCV(
        estimator=base_svc,
        method="sigmoid",
        cv=cv,
    )


def get_voting_ensemble(
    weights: Optional[List[float]] = None,
    nb_alpha: float = 0.1,
    lr_c: float = 1.0,
    svm_c: float = 1.0,
    svm_cv: int = 3,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> VotingClassifier:
    """Soft-voting ensemble of MultinomialNB, LogisticRegression, and Calibrated LinearSVC."""
    nb = get_naive_bayes(alpha=nb_alpha)
    lr = get_logistic_regression(C=lr_c, random_state=random_state)
    svm = get_calibrated_svm(C=svm_c, cv=svm_cv, random_state=random_state)

    estimators = [
        ("nb", nb),
        ("lr", lr),
        ("svm", svm),
    ]

    effective_weights = weights if weights is not None else [1.0, 1.0, 1.0]

    return VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=effective_weights,
    )


def optimize_soft_voting_weights(
    prob_nb: np.ndarray,
    prob_lr: np.ndarray,
    prob_svm: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[List[float], float]:
    """Search for optimal voting weights on validation data using Spam F1 score.

    Evaluates a fine grid of non-negative weights on training cross-validation
    predictions without touching the test partition.
    """
    best_weights = [1.0, 1.0, 1.0]
    best_f1 = -1.0

    # Grid search candidate weight ratios
    grid_steps = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    for w_nb in grid_steps:
        for w_lr in grid_steps:
            for w_svm in grid_steps:
                w_sum = w_nb + w_lr + w_svm
                # Calculate weighted probability for class 1 (spam)
                p_spam = (
                    w_nb * prob_nb[:, 1]
                    + w_lr * prob_lr[:, 1]
                    + w_svm * prob_svm[:, 1]
                ) / w_sum

                preds = (p_spam >= 0.5).astype(int)
                score = f1_score(y_true, preds, pos_label=1, zero_division=0)

                if score > best_f1:
                    best_f1 = score
                    best_weights = [float(w_nb), float(w_lr), float(w_svm)]

    return best_weights, float(best_f1)
